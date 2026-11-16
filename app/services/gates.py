from app.schemas import CandidateProfile, GateResult, JobAnalysis, JobMatchGateResult


def validate_cv_gate(profile: CandidateProfile) -> GateResult:
    missing_sections = [
        section
        for section, values in (
            ("skills", profile.skills),
            ("work experience", profile.work_experience),
            ("projects", profile.projects),
        )
        if not values
    ]
    if len(missing_sections) == 3:
        return GateResult(
            passed=False,
            reason="Insufficient candidate information.",
            warnings=["No skills, experience, or projects were identified."],
        )

    return GateResult(
        passed=True,
        reason="Sufficient candidate information was extracted.",
        warnings=[
            f"No {section} {'was' if section == 'work experience' else 'were'} identified."
            for section in missing_sections
        ],
    )


def evaluate_job_match_gate(analysis: JobAnalysis) -> JobMatchGateResult:
    critical_gap_count = sum(
        requirement.priority == "required"
        and requirement.match_status != "matched"
        for requirement in analysis.requirements
    )
    return JobMatchGateResult(
        passed=critical_gap_count == 0,
        reason=(
            "All required requirements have supporting evidence."
            if critical_gap_count == 0
            else "One or more required requirements need review."
        ),
        warnings=(
            [f"{critical_gap_count} required requirement(s) need review."]
            if critical_gap_count
            else []
        ),
        critical_gap_count=critical_gap_count,
    )
