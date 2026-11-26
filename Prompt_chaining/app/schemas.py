from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalysisRequest(StrictModel):
    cv_text: str = Field(min_length=1, max_length=100_000)
    job_description: str = Field(min_length=1, max_length=100_000)

    @field_validator("cv_text", "job_description")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Input must not be empty.")
        return value


class SkillEvidence(StrictModel):
    name: str
    evidence: list[str]


class WorkExperience(StrictModel):
    job_title: str
    company: str | None
    duration: str | None
    responsibilities: list[str]
    technologies: list[str]
    evidence: list[str]


class Project(StrictModel):
    name: str
    description: str
    technologies: list[str]
    evidence: list[str]


class CandidateProfile(StrictModel):
    candidate_summary: str
    skills: list[SkillEvidence]
    work_experience: list[WorkExperience]
    projects: list[Project]
    education: list[str]
    certifications: list[str]
    achievements: list[str]
    missing_information: list[str]


class RequirementMatch(StrictModel):
    requirement_id: str
    requirement: str
    category: str
    priority: Literal["required", "preferred"]
    match_status: Literal["matched", "partially_matched", "not_found", "unclear"]
    cv_evidence: list[str]
    explanation: str


class JobAnalysis(StrictModel):
    overall_summary: str
    requirements: list[RequirementMatch]
    strengths: list[str]
    areas_to_improve: list[str]


class BulletSuggestion(StrictModel):
    suggested_bullet: str
    based_on_evidence: list[str]


class ApplicationReadyResult(StrictModel):
    match_summary: str
    cover_letter: str
    resume_bullet_suggestions: list[BulletSuggestion]
    interview_topics: list[str]
    improvement_suggestions: list[str]


class GapExplanation(StrictModel):
    requirement: str
    reason: str
    clarification_question: str
    recommendation: str


class GapReviewResult(StrictModel):
    gap_summary: str
    critical_gaps: list[GapExplanation]
    transferable_skills: list[str]
    learning_recommendations: list[str]
    resume_improvement_suggestions: list[str]


class GateResult(StrictModel):
    passed: bool
    reason: str
    warnings: list[str] = Field(default_factory=list)


class JobMatchGateResult(GateResult):
    critical_gap_count: int


class GateResults(StrictModel):
    cv_validation: GateResult
    job_match: JobMatchGateResult | None


class AnalysisResponse(StrictModel):
    status: Literal["ready_to_apply", "needs_review", "needs_input"]
    route: Literal["application_ready", "gap_review"] | None
    candidate_profile: CandidateProfile
    job_analysis: JobAnalysis | None
    gate_results: GateResults
    application_result: ApplicationReadyResult | GapReviewResult | None
