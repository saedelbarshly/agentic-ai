from typing import Any

from pydantic import BaseModel

from app.schemas import (
    ApplicationReadyResult,
    CandidateProfile,
    GapReviewResult,
    JobAnalysis,
    RequirementMatch,
    SkillEvidence,
)

CV_TEXT = "Python developer. Developed Python APIs."
JOB_TEXT = "We need a Python developer with FastAPI."


def profile_with_python() -> CandidateProfile:
    return CandidateProfile(
        candidate_summary="Python developer.",
        skills=[SkillEvidence(name="Python", evidence=["Developed Python APIs."])],
        work_experience=[],
        projects=[],
        education=[],
        certifications=[],
        achievements=[],
        missing_information=[],
    )


def empty_profile() -> CandidateProfile:
    return CandidateProfile(
        candidate_summary="",
        skills=[],
        work_experience=[],
        projects=[],
        education=[],
        certifications=[],
        achievements=[],
        missing_information=[],
    )


def analysis_with(*requirements: RequirementMatch) -> JobAnalysis:
    return JobAnalysis(
        overall_summary="Evidence compared with job requirements.",
        requirements=list(requirements),
        strengths=[],
        areas_to_improve=[],
    )


def requirement(
    *,
    name: str = "Python",
    priority: str = "required",
    status: str = "matched",
    evidence: list[str] | None = None,
) -> RequirementMatch:
    return RequirementMatch(
        requirement_id=name.lower(),
        requirement=name,
        category="technical",
        priority=priority,
        match_status=status,
        cv_evidence=evidence or ["Developed Python APIs."],
        explanation="Evidence was reviewed.",
    )


def ready_result(
    evidence: list[str] | None = None,
) -> ApplicationReadyResult:
    return ApplicationReadyResult(
        match_summary="Relevant documented experience.",
        cover_letter="I am applying for this role.",
        resume_bullet_suggestions=[
            {
                "suggested_bullet": "Developed Python APIs.",
                "based_on_evidence": evidence or ["Developed Python APIs."],
            }
        ],
        interview_topics=["API design"],
        improvement_suggestions=["Clarify project scope."],
    )


def gap_result() -> GapReviewResult:
    return GapReviewResult(
        gap_summary="More evidence is needed.",
        critical_gaps=[
            {
                "requirement": "FastAPI",
                "reason": "The CV does not provide direct evidence.",
                "clarification_question": "Have you used FastAPI?",
                "recommendation": "Add relevant evidence if applicable.",
            }
        ],
        transferable_skills=["Python API development"],
        learning_recommendations=["Review FastAPI basics."],
        resume_improvement_suggestions=["Clarify API experience."],
    )


class FakeProvider:
    def __init__(self, *responses: BaseModel | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, *, system_prompt, payload, response_model):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "payload": payload,
                "response_model": response_model,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, response_model)
        return response
