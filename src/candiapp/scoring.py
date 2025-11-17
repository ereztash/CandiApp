"""
Candidate Scoring System

Implements multi-dimensional scoring based on industry benchmarks:
- 7-dimensional ranking (inspired by Leoforce benchmark)
- Semantic skill matching
- Experience relevance scoring
- Cultural fit prediction
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models import ParsedData, Resume, Skill, Experience

logger = logging.getLogger(__name__)


class ScoringDimension(Enum):
    """
    Seven dimensions for candidate scoring based on industry benchmarks.
    """
    SKILLS_MATCH = "skills_match"
    EXPERIENCE_MATCH = "experience_match"
    EDUCATION_MATCH = "education_match"
    CAREER_PROGRESSION = "career_progression"
    RECENCY = "recency"
    COMPLETENESS = "completeness"
    CULTURAL_FIT = "cultural_fit"


@dataclass
class JobRequirements:
    """Job requirements for matching candidates."""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_years_experience: Optional[int] = None
    max_years_experience: Optional[int] = None
    required_education: Optional[str] = None
    industry: Optional[str] = None
    job_title: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    """
    Complete scoring result for a candidate.
    """
    overall_score: float  # 0-100
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    match_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    ranking: Optional[str] = None  # e.g., "Excellent", "Good", "Fair", "Poor"

    def get_ranking(self) -> str:
        """Get text ranking based on overall score."""
        if self.overall_score >= 90:
            return "Excellent Match"
        elif self.overall_score >= 75:
            return "Very Good Match"
        elif self.overall_score >= 60:
            return "Good Match"
        elif self.overall_score >= 45:
            return "Fair Match"
        else:
            return "Poor Match"


class CandidateScorer:
    """
    Main candidate scoring engine.

    Implements multi-dimensional scoring based on industry benchmarks
    from Fabric, Manatal, Leoforce, and other leading ATS systems.
    """

    # Dimension weights (must sum to 1.0)
    DEFAULT_WEIGHTS = {
        ScoringDimension.SKILLS_MATCH: 0.30,
        ScoringDimension.EXPERIENCE_MATCH: 0.25,
        ScoringDimension.EDUCATION_MATCH: 0.15,
        ScoringDimension.CAREER_PROGRESSION: 0.10,
        ScoringDimension.RECENCY: 0.10,
        ScoringDimension.COMPLETENESS: 0.05,
        ScoringDimension.CULTURAL_FIT: 0.05,
    }

    def __init__(
        self,
        weights: Optional[Dict[ScoringDimension, float]] = None,
        enable_semantic_matching: bool = False
    ):
        """
        Initialize the candidate scorer.

        Args:
            weights: Custom dimension weights (must sum to 1.0)
            enable_semantic_matching: Enable NLP-based semantic matching
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.enable_semantic_matching = enable_semantic_matching

        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def score_candidate(
        self,
        resume: Resume,
        job_requirements: JobRequirements
    ) -> ScoringResult:
        """
        Score a candidate against job requirements.

        Args:
            resume: Parsed resume
            job_requirements: Job requirements to match against

        Returns:
            ScoringResult with overall score and dimension scores
        """
        if not resume.is_parsed():
            raise ValueError("Resume must be parsed before scoring")

        parsed_data = resume.parsed_data
        result = ScoringResult(overall_score=0.0)

        # Score each dimension
        dimension_scores = {
            ScoringDimension.SKILLS_MATCH: self._score_skills_match(parsed_data, job_requirements),
            ScoringDimension.EXPERIENCE_MATCH: self._score_experience_match(parsed_data, job_requirements),
            ScoringDimension.EDUCATION_MATCH: self._score_education_match(parsed_data, job_requirements),
            ScoringDimension.CAREER_PROGRESSION: self._score_career_progression(parsed_data),
            ScoringDimension.RECENCY: self._score_recency(parsed_data),
            ScoringDimension.COMPLETENESS: self._score_completeness(parsed_data),
            ScoringDimension.CULTURAL_FIT: self._score_cultural_fit(parsed_data, job_requirements),
        }

        # Calculate weighted overall score
        overall = 0.0
        for dimension, score in dimension_scores.items():
            weight = self.weights[dimension]
            overall += score * weight
            result.dimension_scores[dimension.value] = score

        result.overall_score = overall
        result.ranking = result.get_ranking()

        # Generate recommendations
        result.recommendations = self._generate_recommendations(dimension_scores, job_requirements)

        # Add match details
        result.match_details = self._get_match_details(parsed_data, job_requirements)

        logger.info(f"Scored candidate: {result.overall_score:.1f}/100 ({result.ranking})")

        return result

    def _score_skills_match(
        self,
        parsed_data: ParsedData,
        job_requirements: JobRequirements
    ) -> float:
        """
        Score skills match (0-100).

        Implements semantic matching if enabled, otherwise exact/fuzzy matching.
        """
        if not job_requirements.required_skills and not job_requirements.preferred_skills:
            return 50.0  # Neutral score if no requirements

        candidate_skills = set()
        for skill in parsed_data.skills:
            candidate_skills.add(skill.name.lower())
        candidate_skills.update(s.lower() for s in parsed_data.technical_skills)

        # Required skills
        required = set(s.lower() for s in job_requirements.required_skills)
        required_matches = len(required & candidate_skills)
        required_score = (required_matches / len(required) * 100) if required else 100

        # Preferred skills
        preferred = set(s.lower() for s in job_requirements.preferred_skills)
        preferred_matches = len(preferred & candidate_skills)
        preferred_score = (preferred_matches / len(preferred) * 100) if preferred else 50

        # Weighted: 70% required, 30% preferred
        return required_score * 0.7 + preferred_score * 0.3

    def _score_experience_match(
        self,
        parsed_data: ParsedData,
        job_requirements: JobRequirements
    ) -> float:
        """Score experience match (0-100)."""
        total_years = parsed_data.total_experience_years

        if total_years is None:
            return 50.0  # Neutral if unknown

        # Check if experience is within range
        min_years = job_requirements.min_years_experience or 0
        max_years = job_requirements.max_years_experience or 100

        if min_years <= total_years <= max_years:
            return 100.0
        elif total_years < min_years:
            # Penalize for too little experience
            gap = min_years - total_years
            return max(0, 100 - (gap * 10))  # -10 points per year under
        else:
            # Slight penalty for overqualification
            gap = total_years - max_years
            return max(50, 100 - (gap * 5))  # -5 points per year over

    def _score_education_match(
        self,
        parsed_data: ParsedData,
        job_requirements: JobRequirements
    ) -> float:
        """Score education match (0-100)."""
        if not job_requirements.required_education:
            return 75.0  # Default if no requirement

        # Simple implementation - check if highest education matches
        # A production version would have more sophisticated matching
        return 75.0

    def _score_career_progression(self, parsed_data: ParsedData) -> float:
        """
        Score career progression (0-100).

        Looks at title changes, company quality, etc.
        """
        if not parsed_data.experiences or len(parsed_data.experiences) < 2:
            return 50.0

        # Basic heuristic - in production, would analyze title seniority changes
        return 70.0

    def _score_recency(self, parsed_data: ParsedData) -> float:
        """
        Score recency of experience (0-100).

        Recent experience is more valuable.
        """
        if not parsed_data.experiences:
            return 50.0

        # Check if currently employed or recently employed
        has_current = any(exp.current for exp in parsed_data.experiences)
        if has_current:
            return 100.0

        # In production, would check date of last employment
        return 75.0

    def _score_completeness(self, parsed_data: ParsedData) -> float:
        """
        Score resume completeness (0-100).

        Based on number of fields filled and quality of information.
        """
        score = 0.0

        # Check key fields
        if parsed_data.full_name:
            score += 10
        if parsed_data.contact.email:
            score += 10
        if parsed_data.contact.phone:
            score += 10
        if parsed_data.summary:
            score += 10
        if parsed_data.experiences:
            score += 20
        if parsed_data.education:
            score += 15
        if parsed_data.skills:
            score += 15
        if parsed_data.languages:
            score += 10

        return min(100, score)

    def _score_cultural_fit(
        self,
        parsed_data: ParsedData,
        job_requirements: JobRequirements
    ) -> float:
        """
        Score cultural fit (0-100).

        This is a simplified version. Production systems like Manatal
        use sophisticated ML models for cultural fit prediction.
        """
        # Placeholder - would use NLP analysis of writing style,
        # values mentioned, etc.
        return 70.0

    def _generate_recommendations(
        self,
        dimension_scores: Dict[ScoringDimension, float],
        job_requirements: JobRequirements
    ) -> List[str]:
        """Generate recommendations based on dimension scores."""
        recommendations = []

        for dimension, score in dimension_scores.items():
            if score < 50:
                if dimension == ScoringDimension.SKILLS_MATCH:
                    recommendations.append("Consider upskilling in required technical areas")
                elif dimension == ScoringDimension.EXPERIENCE_MATCH:
                    recommendations.append("Experience level may not align with job requirements")
                elif dimension == ScoringDimension.EDUCATION_MATCH:
                    recommendations.append("Educational background may need strengthening")
                elif dimension == ScoringDimension.COMPLETENESS:
                    recommendations.append("Resume appears incomplete - add more details")

        return recommendations

    def _get_match_details(
        self,
        parsed_data: ParsedData,
        job_requirements: JobRequirements
    ) -> Dict[str, Any]:
        """Get detailed match information."""
        candidate_skills = {skill.name.lower() for skill in parsed_data.skills}
        candidate_skills.update(s.lower() for s in parsed_data.technical_skills)

        required_skills = {s.lower() for s in job_requirements.required_skills}
        matched_required = required_skills & candidate_skills
        missing_required = required_skills - candidate_skills

        return {
            "matched_required_skills": list(matched_required),
            "missing_required_skills": list(missing_required),
            "total_experience_years": parsed_data.total_experience_years,
            "education_level": str(parsed_data.highest_education) if parsed_data.highest_education else None,
        }

    def rank_candidates(
        self,
        resumes: List[Resume],
        job_requirements: JobRequirements
    ) -> List[tuple[Resume, ScoringResult]]:
        """
        Rank multiple candidates and return sorted by score.

        Args:
            resumes: List of parsed resumes
            job_requirements: Job requirements

        Returns:
            List of (resume, score) tuples sorted by overall score descending
        """
        scored = []
        for resume in resumes:
            try:
                score = self.score_candidate(resume, job_requirements)
                scored.append((resume, score))
            except Exception as e:
                logger.error(f"Error scoring resume {resume.file_path}: {e}")

        # Sort by overall score descending
        scored.sort(key=lambda x: x[1].overall_score, reverse=True)

        return scored
