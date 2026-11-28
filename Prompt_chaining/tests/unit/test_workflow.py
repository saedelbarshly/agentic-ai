import pytest

from app.core.config import Settings
from app.core.errors import ProviderUnavailableError
from app.prompts import APPLICATION_READY_PROMPT, GAP_REVIEW_PROMPT
from app.schemas import (
    AnalysisRequest,
    ApplicationReadyResult,
    CandidateProfile,
    GapReviewResult,
    JobAnalysis,
    RequirementMatch,
    SkillEvidence,
)
from app.services.workflow import AnalysisWorkflow
from tests.helpers import (
    CV_TEXT,
    JOB_TEXT,
    FakeProvider,
    analysis_with,
    empty_profile,
    gap_result,
    profile_with_python,
    ready_result,
    requirement,
)


def request() -> AnalysisRequest:
    return AnalysisRequest(cv_text=CV_TEXT, job_description=JOB_TEXT)


def workflow(provider: FakeProvider) -> AnalysisWorkflow:
    return AnalysisWorkflow(provider, Settings(_env_file=None))


def test_cv_gate_failure_stops_before_job_analysis():
    provider = FakeProvider(empty_profile())

    result = workflow(provider).run(request(), "request-1")

    assert result.status == "needs_input"
    assert result.route is None
    assert result.job_analysis is None
    assert result.application_result is None
    assert len(provider.calls) == 1


def test_job_analysis_receives_profile_and_original_job_description():
    provider = FakeProvider(
        profile_with_python(),
        analysis_with(requirement()),
        ready_result(),
    )

    workflow(provider).run(request(), "request-2")

    job_call = provider.calls[1]
    assert job_call["payload"]["job_description"] == JOB_TEXT
    assert job_call["payload"]["candidate_profile"]["skills"][0]["name"] == "Python"
    assert job_call["response_model"] is JobAnalysis


def test_application_ready_route_uses_ready_prompt_and_schema():
    provider = FakeProvider(
        profile_with_python(),
        analysis_with(requirement()),
        ready_result(),
    )

    result = workflow(provider).run(request(), "request-3")

    assert result.status == "ready_to_apply"
    assert result.route == "application_ready"
    assert isinstance(result.application_result, ApplicationReadyResult)
    assert provider.calls[2]["system_prompt"] == APPLICATION_READY_PROMPT
    assert provider.calls[2]["response_model"] is ApplicationReadyResult


def test_gap_route_uses_gap_prompt_and_schema():
    provider = FakeProvider(
        profile_with_python(),
        analysis_with(requirement(name="FastAPI", status="partially_matched")),
        gap_result(),
    )

    result = workflow(provider).run(request(), "request-4")

    assert result.status == "needs_review"
    assert result.route == "gap_review"
    assert isinstance(result.application_result, GapReviewResult)
    assert provider.calls[2]["system_prompt"] == GAP_REVIEW_PROMPT
    assert provider.calls[2]["response_model"] is GapReviewResult


def test_unsupported_profile_evidence_is_removed_and_cannot_match():
    fabricated_profile = profile_with_python().model_copy(
        update={
            "skills": [
                SkillEvidence(name="Python", evidence=["Developed Python APIs."]),
                SkillEvidence(name="Rust", evidence=["Expert Rust engineer."]),
            ]
        }
    )
    fabricated_analysis = analysis_with(
        RequirementMatch(
            requirement_id="rust",
            requirement="Rust",
            category="technical",
            priority="required",
            match_status="matched",
            cv_evidence=["Expert Rust engineer."],
            explanation="The candidate is an expert.",
        )
    )
    provider = FakeProvider(fabricated_profile, fabricated_analysis, gap_result())

    result = workflow(provider).run(request(), "request-5")

    assert [skill.name for skill in result.candidate_profile.skills] == ["Python"]
    assert result.job_analysis.requirements[0].match_status == "not_found"
    assert result.job_analysis.requirements[0].cv_evidence == []
    assert result.route == "gap_review"
    assert result.application_result.critical_gaps[0].requirement == "Rust"
    assert "does not establish" in result.application_result.critical_gaps[0].reason


def test_unsupported_bullet_evidence_is_not_returned():
    provider = FakeProvider(
        profile_with_python(),
        analysis_with(requirement()),
        ready_result(["unverified achievement"]),
    )

    result = workflow(provider).run(request(), "request-6")

    assert isinstance(result.application_result, ApplicationReadyResult)
    assert result.application_result.resume_bullet_suggestions == []


def test_provider_failure_stops_workflow():
    provider = FakeProvider(profile_with_python(), ProviderUnavailableError())

    with pytest.raises(ProviderUnavailableError):
        workflow(provider).run(request(), "request-7")

    assert len(provider.calls) == 2
