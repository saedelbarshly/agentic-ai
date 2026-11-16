import logging
import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from app.core.config import Settings
from app.prompts import (
    APPLICATION_READY_PROMPT,
    CV_EXTRACTION_PROMPT,
    GAP_REVIEW_PROMPT,
    JOB_ANALYSIS_PROMPT,
)
from app.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    ApplicationReadyResult,
    CandidateProfile,
    GapExplanation,
    GapReviewResult,
    GateResults,
    JobAnalysis,
    JobMatchGateResult,
)
from app.services.gates import evaluate_job_match_gate, validate_cv_gate

logger = logging.getLogger(__name__)
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class JSONProvider(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[ResponseModel],
    ) -> ResponseModel: ...


class AnalysisWorkflow:
    def __init__(self, provider: JSONProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    def run(self, request: AnalysisRequest, request_id: str) -> AnalysisResponse:
        if len(request.cv_text) > self._settings.max_cv_chars:
            raise ValueError("CV exceeds the configured character limit.")
        if len(request.job_description) > self._settings.max_job_description_chars:
            raise ValueError("Job description exceeds the configured character limit.")

        candidate_profile = self._run_stage(
            "cv_extraction",
            request_id,
            CV_EXTRACTION_PROMPT,
            {"cv_text": request.cv_text},
            CandidateProfile,
        )
        candidate_profile = self._validate_candidate_evidence(
            candidate_profile, request.cv_text
        )
        cv_gate = validate_cv_gate(candidate_profile)
        if not cv_gate.passed:
            return AnalysisResponse(
                status="needs_input",
                route=None,
                candidate_profile=candidate_profile,
                job_analysis=None,
                gate_results=GateResults(cv_validation=cv_gate, job_match=None),
                application_result=None,
            )

        job_analysis = self._run_stage(
            "job_analysis",
            request_id,
            JOB_ANALYSIS_PROMPT,
            {
                "candidate_profile": candidate_profile.model_dump(mode="json"),
                "job_description": request.job_description,
            },
            JobAnalysis,
        )
        job_analysis = self._validate_job_evidence(job_analysis, candidate_profile)
        job_gate = evaluate_job_match_gate(job_analysis)

        if job_gate.passed:
            application_result = self._run_stage(
                "application_ready",
                request_id,
                APPLICATION_READY_PROMPT,
                {
                    "candidate_profile": candidate_profile.model_dump(mode="json"),
                    "job_description": request.job_description,
                    "job_analysis": job_analysis.model_dump(mode="json"),
                },
                ApplicationReadyResult,
            )
            application_result = self._validate_bullet_evidence(
                application_result, candidate_profile
            )
            route = "application_ready"
            status = "ready_to_apply"
        else:
            application_result = self._run_stage(
                "gap_review",
                request_id,
                GAP_REVIEW_PROMPT,
                {
                    "candidate_profile": candidate_profile.model_dump(mode="json"),
                    "job_description": request.job_description,
                    "job_analysis": job_analysis.model_dump(mode="json"),
                },
                GapReviewResult,
            )
            application_result = self._validate_gap_result(
                application_result, job_analysis, candidate_profile
            )
            route = "gap_review"
            status = "needs_review"

        return AnalysisResponse(
            status=status,
            route=route,
            candidate_profile=candidate_profile,
            job_analysis=job_analysis,
            gate_results=GateResults(cv_validation=cv_gate, job_match=job_gate),
            application_result=application_result,
        )

    def _run_stage(
        self,
        stage: str,
        request_id: str,
        prompt: str,
        payload: dict[str, Any],
        response_model: type[ResponseModel],
    ) -> ResponseModel:
        started = time.monotonic()
        try:
            result = self._provider.complete_json(
                system_prompt=prompt,
                payload=payload,
                response_model=response_model,
            )
        except Exception as exc:
            logger.error(
                "workflow_stage_failed request_id=%s stage=%s error_type=%s",
                request_id,
                stage,
                type(exc).__name__,
            )
            raise
        logger.info(
            "workflow_stage_succeeded request_id=%s stage=%s duration_ms=%.2f",
            request_id,
            stage,
            (time.monotonic() - started) * 1000,
        )
        return result

    @classmethod
    def _validate_candidate_evidence(
        cls, profile: CandidateProfile, cv_text: str
    ) -> CandidateProfile:
        def exact_items(items: list[str]) -> list[str]:
            return [item for item in items if cls._is_excerpt(item, cv_text)]

        skills = [
            skill.model_copy(update={"evidence": exact_items(skill.evidence)})
            for skill in profile.skills
            if cls._is_excerpt(skill.name, cv_text)
            and exact_items(skill.evidence)
        ]
        experiences = [
            experience.model_copy(
                update={
                    "company": (
                        experience.company
                        if experience.company and cls._is_excerpt(experience.company, cv_text)
                        else None
                    ),
                    "duration": (
                        experience.duration
                        if experience.duration and cls._is_excerpt(experience.duration, cv_text)
                        else None
                    ),
                    "responsibilities": exact_items(experience.responsibilities),
                    "technologies": exact_items(experience.technologies),
                    "evidence": exact_items(experience.evidence),
                }
            )
            for experience in profile.work_experience
            if cls._is_excerpt(experience.job_title, cv_text)
            and exact_items(experience.evidence)
        ]
        projects = [
            project.model_copy(
                update={
                    "description": (
                        project.description
                        if cls._is_excerpt(project.description, cv_text)
                        else ""
                    ),
                    "technologies": exact_items(project.technologies),
                    "evidence": exact_items(project.evidence),
                }
            )
            for project in profile.projects
            if cls._is_excerpt(project.name, cv_text)
            and exact_items(project.evidence)
        ]
        return profile.model_copy(
            update={
                "candidate_summary": (
                    profile.candidate_summary
                    if cls._is_excerpt(profile.candidate_summary, cv_text)
                    else ""
                ),
                "skills": skills,
                "work_experience": experiences,
                "projects": projects,
                "education": exact_items(profile.education),
                "certifications": exact_items(profile.certifications),
                "achievements": exact_items(profile.achievements),
            }
        )

    @classmethod
    def _validate_job_evidence(
        cls, analysis: JobAnalysis, profile: CandidateProfile
    ) -> JobAnalysis:
        allowed_evidence = {
            cls._normalize(evidence)
            for evidence in cls._candidate_evidence(profile)
        }
        requirements = []
        for requirement in analysis.requirements:
            evidence = [
                item
                for item in requirement.cv_evidence
                if cls._normalize(item) in allowed_evidence
            ]
            match_status = requirement.match_status
            explanation = requirement.explanation
            if match_status == "not_found":
                evidence = []
            if match_status in ("matched", "partially_matched") and not evidence:
                match_status = "not_found"
            if not evidence:
                explanation = (
                    "The CV provides no validated evidence for this requirement; "
                    "this does not establish whether the candidate has the skill."
                )
            requirements.append(
                requirement.model_copy(
                    update={
                        "cv_evidence": evidence,
                        "match_status": match_status,
                        "explanation": explanation,
                    }
                )
            )
        return analysis.model_copy(update={"requirements": requirements})

    @classmethod
    def _validate_bullet_evidence(
        cls, result: ApplicationReadyResult, profile: CandidateProfile
    ) -> ApplicationReadyResult:
        allowed_evidence = {
            cls._normalize(evidence) for evidence in cls._candidate_evidence(profile)
        }
        bullets = [
            bullet.model_copy(
                update={
                    "based_on_evidence": [
                        evidence
                        for evidence in bullet.based_on_evidence
                        if cls._normalize(evidence) in allowed_evidence
                    ]
                }
            )
            for bullet in result.resume_bullet_suggestions
            if any(
                cls._normalize(evidence) in allowed_evidence
                for evidence in bullet.based_on_evidence
            )
        ]
        return result.model_copy(update={"resume_bullet_suggestions": bullets})

    @classmethod
    def _validate_gap_result(
        cls,
        result: GapReviewResult,
        analysis: JobAnalysis,
        profile: CandidateProfile,
    ) -> GapReviewResult:
        critical_requirements = [
            requirement
            for requirement in analysis.requirements
            if requirement.priority == "required"
            and requirement.match_status != "matched"
        ]
        generated_gaps = {
            cls._normalize(gap.requirement): gap for gap in result.critical_gaps
        }
        validated_gaps = []
        for requirement in critical_requirements:
            generated = generated_gaps.get(cls._normalize(requirement.requirement))
            if requirement.match_status == "partially_matched":
                reason = "The available CV evidence only partially demonstrates this requirement."
            elif requirement.match_status == "unclear":
                reason = "The available CV information is insufficient to assess this requirement."
            else:
                reason = (
                    "The CV provides no evidence for this requirement; this does not establish "
                    "whether the candidate has the skill."
                )
            validated_gaps.append(
                GapExplanation(
                    requirement=requirement.requirement,
                    reason=reason,
                    clarification_question=(
                        generated.clarification_question
                        if generated
                        else f"Can you share a specific example related to {requirement.requirement}?"
                    ),
                    recommendation=(
                        generated.recommendation
                        if generated
                        else "If applicable, add truthful CV evidence for this requirement."
                    ),
                )
            )

        supported_transferable_skills = {
            cls._normalize(skill.name) for skill in profile.skills
        }
        supported_transferable_skills.update(
            cls._normalize(item)
            for experience in profile.work_experience
            for item in experience.technologies + experience.responsibilities
        )
        supported_transferable_skills.update(
            cls._normalize(item)
            for project in profile.projects
            for item in project.technologies
        )
        return result.model_copy(
            update={
                "critical_gaps": validated_gaps,
                "transferable_skills": [
                    skill
                    for skill in result.transferable_skills
                    if cls._normalize(skill) in supported_transferable_skills
                ],
            }
        )

    @staticmethod
    def _candidate_evidence(profile: CandidateProfile) -> list[str]:
        evidence = [item for skill in profile.skills for item in skill.evidence]
        evidence.extend(
            item for experience in profile.work_experience for item in experience.evidence
        )
        evidence.extend(item for project in profile.projects for item in project.evidence)
        return evidence

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    @classmethod
    def _is_excerpt(cls, item: str, original: str) -> bool:
        return bool(item.strip()) and cls._normalize(item) in cls._normalize(original)
