"""
Resume Screening Engine
Main orchestrator that combines all engines to screen resumes.
"""
import time
from typing import Tuple, Optional
from uuid import UUID
import logging

from models.archetype import ProcessArchetype, ArchetypeProfile
from models.job import JobRequirement, ArchetypeFitCriteria, SkillMatchCriteria
from models.candidate import (
    ScreeningResult,
    ScreeningDecision,
    RejectionReason,
    ProcessFitScore,
    SemanticFitScore
)
from engines.verb_extractor import VerbExtractor
from engines.hydrodynamic import HydrodynamicController, ComplexityMetrics
from engines.semantic_matcher import SemanticMatcher
from config.settings import settings

logger = logging.getLogger(__name__)


class ResumeScreeningEngine:
    """
    Main screening engine that orchestrates the entire screening process.
    Combines process-fit analysis, semantic matching, and hydrodynamic control.
    """

    def __init__(
        self,
        verb_extractor: Optional[VerbExtractor] = None,
        hydrodynamic_controller: Optional[HydrodynamicController] = None,
        semantic_matcher: Optional[SemanticMatcher] = None
    ):
        """
        Initialize the screening engine.

        Args:
            verb_extractor: VerbExtractor instance (creates new if None)
            hydrodynamic_controller: HydrodynamicController instance (creates new if None)
            semantic_matcher: SemanticMatcher instance (creates new if None)
        """
        self.verb_extractor = verb_extractor or VerbExtractor()
        self.hydrodynamic_controller = hydrodynamic_controller or HydrodynamicController(
            reynolds_threshold_low=settings.REYNOLDS_THRESHOLD_LOW,
            reynolds_threshold_high=settings.REYNOLDS_THRESHOLD_HIGH
        )
        self.semantic_matcher = semantic_matcher or SemanticMatcher()

        self.process_fit_weight = settings.PROCESS_FIT_WEIGHT
        self.semantic_fit_weight = settings.SEMANTIC_FIT_WEIGHT
        self.pass_threshold = settings.PASS_THRESHOLD

        logger.info("ResumeScreeningEngine initialized successfully")

    def screen_resume(
        self,
        candidate_id: UUID,
        job_id: UUID,
        resume_text: str,
        job: JobRequirement
    ) -> ScreeningResult:
        """
        Screen a resume against job requirements.

        Args:
            candidate_id: Candidate UUID
            job_id: Job UUID
            resume_text: Full resume text
            job: Job requirement details

        Returns:
            ScreeningResult with complete analysis
        """
        start_time = time.time()

        logger.info(f"Starting screening for candidate {candidate_id} for job {job_id}")

        # Step 1: Hydrodynamic analysis (optional optimization)
        complexity_metrics = None
        if settings.ENABLE_HYDRODYNAMIC_CONTROL:
            use_deep, complexity_metrics = self.hydrodynamic_controller.should_use_deep_processing(
                resume_text
            )
            logger.info(
                f"Hydrodynamic analysis: Re={complexity_metrics.reynolds_number:.2f}, "
                f"Deep={use_deep}"
            )

        # Step 2: Process-Fit Analysis
        process_fit_score = self._analyze_process_fit(
            resume_text=resume_text,
            job=job
        )

        # Step 3: Semantic/Skills Analysis
        semantic_fit_score = self._analyze_semantic_fit(
            resume_text=resume_text,
            job=job
        )

        # Step 4: Calculate overall score
        overall_score = (
            process_fit_score.overall_score * self.process_fit_weight +
            semantic_fit_score.overall_score * self.semantic_fit_weight
        )

        # Step 5: Make decision
        decision, rejection_reason, rejection_details = self._make_decision(
            overall_score=overall_score,
            process_fit_score=process_fit_score,
            semantic_fit_score=semantic_fit_score,
            job=job
        )

        # Step 6: Generate recommendation
        recommendation = self._generate_recommendation(
            decision=decision,
            overall_score=overall_score,
            process_fit_score=process_fit_score,
            semantic_fit_score=semantic_fit_score,
            job=job
        )

        # Get evidence verbs for detected archetype
        evidence_verbs, _ = self.verb_extractor.get_archetype_evidence(
            profile=process_fit_score.archetype_profile,
            archetype=process_fit_score.archetype_profile.primary_archetype
        )

        # Combine matched skills
        all_matched_skills = (
            semantic_fit_score.matched_required_skills +
            semantic_fit_score.matched_preferred_skills
        )

        # Create screening result
        result = ScreeningResult(
            candidate_id=candidate_id,
            job_id=job_id,
            decision=decision,
            overall_score=overall_score,
            process_fit_score=process_fit_score.overall_score,
            semantic_fit_score=semantic_fit_score.overall_score,
            archetype_detected=process_fit_score.archetype_profile.primary_archetype,
            archetype_confidence=process_fit_score.archetype_profile.primary_confidence,
            archetype_alignment=process_fit_score.archetype_alignment,
            evidence_verbs=evidence_verbs[:10],  # Limit to top 10
            matched_skills=all_matched_skills,
            missing_skills=semantic_fit_score.missing_required_skills,
            rejection_reason=rejection_reason,
            rejection_details=rejection_details,
            recommendation=recommendation
        )

        elapsed_time = time.time() - start_time
        logger.info(
            f"Screening completed in {elapsed_time:.2f}s: "
            f"Decision={decision.value}, Score={overall_score:.2%}"
        )

        return result

    def _analyze_process_fit(
        self,
        resume_text: str,
        job: JobRequirement
    ) -> ProcessFitScore:
        """
        Analyze process-fit based on archetype matching.

        Args:
            resume_text: Resume content
            job: Job requirements

        Returns:
            ProcessFitScore with archetype analysis
        """
        # Generate archetype profile
        profile = self.verb_extractor.profile_resume(resume_text)

        # Calculate archetype match score
        archetype_match_score = self._calculate_archetype_match(
            detected_archetype=profile.primary_archetype,
            detected_score=profile.primary_score,
            required_primary=job.archetype_primary,
            required_secondary=job.archetype_secondary
        )

        # Determine alignment level
        if profile.primary_archetype == job.archetype_primary:
            alignment = "perfect"
        elif job.archetype_secondary and profile.primary_archetype == job.archetype_secondary:
            alignment = "good"
        elif profile.secondary_archetype == job.archetype_primary:
            alignment = "partial"
        else:
            alignment = "mismatch"

        # Calculate evidence strength
        evidence_strength = min(profile.primary_confidence, 1.0)

        # Overall process fit score
        overall_score = archetype_match_score * evidence_strength

        return ProcessFitScore(
            overall_score=overall_score,
            archetype_profile=profile,
            archetype_match_score=archetype_match_score,
            archetype_alignment=alignment,
            evidence_strength=evidence_strength
        )

    def _calculate_archetype_match(
        self,
        detected_archetype: ProcessArchetype,
        detected_score: float,
        required_primary: ProcessArchetype,
        required_secondary: Optional[ProcessArchetype]
    ) -> float:
        """
        Calculate how well detected archetype matches requirements.

        Args:
            detected_archetype: Detected primary archetype
            detected_score: Score for detected archetype
            required_primary: Required primary archetype
            required_secondary: Optional secondary archetype

        Returns:
            Match score (0.0-1.0)
        """
        if detected_archetype == required_primary:
            # Perfect match with primary
            return 1.0
        elif required_secondary and detected_archetype == required_secondary:
            # Match with secondary (good but not perfect)
            return 0.8
        else:
            # Mismatch - give partial credit based on score
            # Even wrong archetype can score if weakly detected
            return detected_score * 0.3

    def _analyze_semantic_fit(
        self,
        resume_text: str,
        job: JobRequirement
    ) -> SemanticFitScore:
        """
        Analyze semantic/skills fit.

        Args:
            resume_text: Resume content
            job: Job requirements

        Returns:
            SemanticFitScore with skills analysis
        """
        return self.semantic_matcher.match_skills(
            resume_text=resume_text,
            required_skills=job.required_skills,
            preferred_skills=job.preferred_skills,
            required_weight=0.8,
            preferred_weight=0.2
        )

    def _make_decision(
        self,
        overall_score: float,
        process_fit_score: ProcessFitScore,
        semantic_fit_score: SemanticFitScore,
        job: JobRequirement
    ) -> Tuple[ScreeningDecision, Optional[RejectionReason], Optional[str]]:
        """
        Make final screening decision.

        Args:
            overall_score: Combined overall score
            process_fit_score: Process fit analysis
            semantic_fit_score: Semantic fit analysis
            job: Job requirements

        Returns:
            Tuple of (decision, rejection_reason, rejection_details)
        """
        # Check overall threshold
        if overall_score < self.pass_threshold:
            # Failed - determine why
            if process_fit_score.archetype_alignment == "mismatch":
                return (
                    ScreeningDecision.FAILED,
                    RejectionReason.ARCHETYPE_MISMATCH,
                    f"Candidate shows {process_fit_score.archetype_profile.primary_archetype.value} "
                    f"traits, but job requires {job.archetype_primary.value}"
                )
            elif semantic_fit_score.required_match_rate < settings.MIN_REQUIRED_SKILLS_MATCH_RATE:
                return (
                    ScreeningDecision.FAILED,
                    RejectionReason.INSUFFICIENT_SKILLS,
                    f"Missing {len(semantic_fit_score.missing_required_skills)} "
                    f"required skills: {', '.join(semantic_fit_score.missing_required_skills[:3])}"
                )
            elif process_fit_score.overall_score < 0.5:
                return (
                    ScreeningDecision.FAILED,
                    RejectionReason.LOW_PROCESS_FIT,
                    f"Process fit score too low: {process_fit_score.overall_score:.2%}"
                )
            else:
                return (
                    ScreeningDecision.FAILED,
                    RejectionReason.LOW_OVERALL_SCORE,
                    f"Overall score below threshold: {overall_score:.2%} < {self.pass_threshold:.2%}"
                )

        # Additional checks even if overall score passes
        if process_fit_score.archetype_profile.primary_archetype != job.archetype_primary:
            # Archetype mismatch - require higher overall score
            if overall_score < 0.8:
                return (
                    ScreeningDecision.FAILED,
                    RejectionReason.ARCHETYPE_MISMATCH,
                    "Archetype mismatch requires higher overall score"
                )

        # Check minimum required skills match
        if semantic_fit_score.required_match_rate < settings.MIN_REQUIRED_SKILLS_MATCH_RATE:
            return (
                ScreeningDecision.FAILED,
                RejectionReason.INSUFFICIENT_SKILLS,
                f"Only {semantic_fit_score.required_match_rate:.2%} of required skills matched"
            )

        # Passed all checks
        return ScreeningDecision.PASSED, None, None

    def _generate_recommendation(
        self,
        decision: ScreeningDecision,
        overall_score: float,
        process_fit_score: ProcessFitScore,
        semantic_fit_score: SemanticFitScore,
        job: JobRequirement
    ) -> str:
        """
        Generate human-readable recommendation for HR.

        Args:
            decision: Screening decision
            overall_score: Overall score
            process_fit_score: Process fit score
            semantic_fit_score: Semantic fit score
            job: Job requirements

        Returns:
            Recommendation text
        """
        if decision == ScreeningDecision.PASSED:
            if overall_score >= 0.85:
                recommendation = "🌟 Strong candidate - highly recommend interview. "
            else:
                recommendation = "✓ Good candidate - recommend interview. "

            recommendation += (
                f"Process fit: {process_fit_score.archetype_alignment} "
                f"({process_fit_score.overall_score:.0%}). "
            )

            if semantic_fit_score.required_match_rate == 1.0:
                recommendation += "All required skills matched. "
            else:
                recommendation += (
                    f"{semantic_fit_score.required_match_rate:.0%} "
                    f"of required skills matched. "
                )

            if process_fit_score.archetype_profile.secondary_archetype:
                recommendation += (
                    f"Also shows {process_fit_score.archetype_profile.secondary_archetype.value} traits. "
                )

        else:
            recommendation = "Not recommended. "

            if process_fit_score.archetype_alignment == "mismatch":
                recommendation += (
                    f"Candidate's {process_fit_score.archetype_profile.primary_archetype.value} "
                    f"profile doesn't align with {job.archetype_primary.value} role. "
                )

            missing_count = len(semantic_fit_score.missing_required_skills)
            if missing_count > 0:
                recommendation += (
                    f"Missing {missing_count} key skills. "
                )

        return recommendation
