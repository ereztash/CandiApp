"""
Unit tests for CandidateScorer
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candiapp.scoring import CandidateScorer, JobRequirements, ScoringDimension, ScoringResult
from candiapp.models import Resume, ParsedData, Skill, Experience


class TestCandidateScorer:
    """Test CandidateScorer functionality."""

    def test_scorer_initialization(self):
        """Test scorer can be initialized."""
        scorer = CandidateScorer()
        assert scorer is not None
        assert scorer.weights is not None

    def test_default_weights_sum_to_one(self):
        """Test default weights sum to 1.0."""
        scorer = CandidateScorer()
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_custom_weights(self):
        """Test custom weights."""
        custom_weights = {
            ScoringDimension.SKILLS_MATCH: 0.40,
            ScoringDimension.EXPERIENCE_MATCH: 0.30,
            ScoringDimension.EDUCATION_MATCH: 0.10,
            ScoringDimension.CAREER_PROGRESSION: 0.05,
            ScoringDimension.RECENCY: 0.05,
            ScoringDimension.COMPLETENESS: 0.05,
            ScoringDimension.CULTURAL_FIT: 0.05,
        }

        scorer = CandidateScorer(weights=custom_weights)
        assert scorer.weights[ScoringDimension.SKILLS_MATCH] == 0.40

    def test_invalid_weights_raise_error(self):
        """Test that invalid weights raise ValueError."""
        invalid_weights = {
            ScoringDimension.SKILLS_MATCH: 0.50,
            ScoringDimension.EXPERIENCE_MATCH: 0.30,
            ScoringDimension.EDUCATION_MATCH: 0.10,
            ScoringDimension.CAREER_PROGRESSION: 0.05,
            ScoringDimension.RECENCY: 0.05,
            ScoringDimension.COMPLETENESS: 0.05,
            ScoringDimension.CULTURAL_FIT: 0.05,
        }

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            CandidateScorer(weights=invalid_weights)

    def test_score_unparsed_resume_raises_error(self):
        """Test that scoring an unparsed resume raises ValueError."""
        scorer = CandidateScorer()
        resume = Resume(file_path="test.pdf", file_type="pdf", file_size=1000)
        job_req = JobRequirements()

        with pytest.raises(ValueError, match="Resume must be parsed"):
            scorer.score_candidate(resume, job_req)

    def test_score_candidate_basic(self):
        """Test basic candidate scoring."""
        scorer = CandidateScorer()

        # Create a resume with parsed data
        resume = Resume(file_path="test.pdf", file_type="pdf", file_size=1000)
        resume.parsed_data = ParsedData()
        resume.parsed_data.full_name = "John Doe"
        resume.parsed_data.contact.email = "john@example.com"
        resume.parsed_data.skills = [
            Skill(name="Python"),
            Skill(name="Django"),
            Skill(name="PostgreSQL"),
        ]
        resume.parsed_data.total_experience_years = 5.0

        # Create job requirements
        job_req = JobRequirements(
            required_skills=["Python", "Django"],
            preferred_skills=["AWS", "Docker"],
            min_years_experience=3,
            max_years_experience=7,
        )

        # Score the candidate
        result = scorer.score_candidate(resume, job_req)

        assert isinstance(result, ScoringResult)
        assert 0 <= result.overall_score <= 100
        assert result.ranking is not None
        assert len(result.dimension_scores) == 7

    def test_skills_match_scoring(self):
        """Test skills match scoring."""
        scorer = CandidateScorer()

        parsed_data = ParsedData()
        parsed_data.skills = [
            Skill(name="Python"),
            Skill(name="Django"),
            Skill(name="PostgreSQL"),
        ]

        # All required skills match
        job_req = JobRequirements(required_skills=["Python", "Django"])
        score = scorer._score_skills_match(parsed_data, job_req)
        assert score >= 85  # Should be high score

        # Partial match
        job_req = JobRequirements(required_skills=["Python", "Django", "AWS", "Docker"])
        score = scorer._score_skills_match(parsed_data, job_req)
        assert 30 <= score <= 70  # Should be moderate score

        # No requirements
        job_req = JobRequirements()
        score = scorer._score_skills_match(parsed_data, job_req)
        assert score == 50.0  # Neutral

    def test_experience_match_scoring(self):
        """Test experience match scoring."""
        scorer = CandidateScorer()

        # Perfect match
        parsed_data = ParsedData()
        parsed_data.total_experience_years = 5.0

        job_req = JobRequirements(min_years_experience=3, max_years_experience=7)
        score = scorer._score_experience_match(parsed_data, job_req)
        assert score == 100.0

        # Under-qualified
        parsed_data.total_experience_years = 1.0
        score = scorer._score_experience_match(parsed_data, job_req)
        assert score < 100.0

        # Over-qualified
        parsed_data.total_experience_years = 15.0
        score = scorer._score_experience_match(parsed_data, job_req)
        assert score < 100.0

    def test_completeness_scoring(self):
        """Test completeness scoring."""
        scorer = CandidateScorer()

        # Minimal data
        parsed_data = ParsedData()
        score = scorer._score_completeness(parsed_data)
        assert score < 50

        # Complete data
        parsed_data.full_name = "John Doe"
        parsed_data.contact.email = "john@example.com"
        parsed_data.contact.phone = "123-456-7890"
        parsed_data.summary = "Experienced developer"
        parsed_data.experiences = [Experience(company="Acme", title="Developer")]
        parsed_data.education = []
        parsed_data.skills = [Skill(name="Python")]
        parsed_data.languages = []

        score = scorer._score_completeness(parsed_data)
        assert score >= 50

    def test_ranking_based_on_score(self):
        """Test ranking determination."""
        result = ScoringResult(overall_score=95)
        assert result.get_ranking() == "Excellent Match"

        result = ScoringResult(overall_score=80)
        assert result.get_ranking() == "Very Good Match"

        result = ScoringResult(overall_score=65)
        assert result.get_ranking() == "Good Match"

        result = ScoringResult(overall_score=50)
        assert result.get_ranking() == "Fair Match"

        result = ScoringResult(overall_score=30)
        assert result.get_ranking() == "Poor Match"

    def test_rank_candidates(self):
        """Test ranking multiple candidates."""
        scorer = CandidateScorer()

        # Create multiple resumes with varying scores
        resumes = []
        for i in range(3):
            resume = Resume(file_path=f"test{i}.pdf", file_type="pdf", file_size=1000)
            resume.parsed_data = ParsedData()
            resume.parsed_data.full_name = f"Candidate {i}"
            resume.parsed_data.contact.email = f"candidate{i}@example.com"
            resume.parsed_data.skills = [Skill(name="Python")] if i > 0 else []
            resume.parsed_data.total_experience_years = float(i * 3)
            resumes.append(resume)

        job_req = JobRequirements(
            required_skills=["Python"],
            min_years_experience=2,
            max_years_experience=6,
        )

        # Rank candidates
        ranked = scorer.rank_candidates(resumes, job_req)

        assert len(ranked) == 3
        # Check that they are sorted by score (descending)
        scores = [result.overall_score for _, result in ranked]
        assert scores == sorted(scores, reverse=True)


class TestJobRequirements:
    """Test JobRequirements model."""

    def test_job_requirements_initialization(self):
        """Test JobRequirements can be initialized."""
        req = JobRequirements()
        assert isinstance(req.required_skills, list)
        assert isinstance(req.preferred_skills, list)
        assert len(req.required_skills) == 0

    def test_job_requirements_with_data(self):
        """Test JobRequirements with data."""
        req = JobRequirements(
            required_skills=["Python", "Django"],
            preferred_skills=["AWS"],
            min_years_experience=3,
            max_years_experience=7,
            job_title="Senior Developer",
        )

        assert len(req.required_skills) == 2
        assert "Python" in req.required_skills
        assert req.min_years_experience == 3
        assert req.job_title == "Senior Developer"
