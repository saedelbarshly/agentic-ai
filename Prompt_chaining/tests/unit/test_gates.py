import pytest

from app.services.gates import evaluate_job_match_gate, validate_cv_gate
from tests.helpers import analysis_with, empty_profile, profile_with_python, requirement
from app.schemas import Project


def test_empty_candidate_profile_fails_cv_gate():
    result = validate_cv_gate(empty_profile())

    assert result.passed is False
    assert result.reason == "Insufficient candidate information."
    assert result.warnings == ["No skills, experience, or projects were identified."]


def test_profile_with_skills_passes_cv_gate():
    result = validate_cv_gate(profile_with_python())

    assert result.passed is True
    assert result.warnings == ["No work experience was identified.", "No projects were identified."]


def test_profile_with_project_and_no_work_experience_passes_cv_gate():
    profile = empty_profile().model_copy(
        update={
            "projects": [
                Project(
                    name="API project",
                    description="A small API",
                    technologies=["Python"],
                    evidence=["Built an API project."],
                )
            ]
        }
    )

    assert validate_cv_gate(profile).passed is True


def test_all_required_requirements_matched_routes_ready():
    gate = evaluate_job_match_gate(analysis_with(requirement()))

    assert gate.passed is True
    assert gate.critical_gap_count == 0


@pytest.mark.parametrize("status", ["not_found", "partially_matched", "unclear"])
def test_unmatched_required_requirement_is_critical(status):
    gate = evaluate_job_match_gate(analysis_with(requirement(status=status)))

    assert gate.passed is False
    assert gate.critical_gap_count == 1


def test_missing_preferred_requirement_is_not_critical():
    gate = evaluate_job_match_gate(
        analysis_with(requirement(priority="preferred", status="not_found", evidence=[]))
    )

    assert gate.passed is True
    assert gate.critical_gap_count == 0
