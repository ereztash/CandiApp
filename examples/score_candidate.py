#!/usr/bin/env python3
"""
Example: Candidate Scoring

This example demonstrates how to score a candidate against job requirements.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candiapp import ResumeParser, CandidateScorer
from candiapp.scoring import JobRequirements


def main():
    """Score a candidate against job requirements."""
    if len(sys.argv) < 2:
        print("Usage: python score_candidate.py <resume_file>")
        print("\nExample:")
        print("  python score_candidate.py sample_resume.pdf")
        sys.exit(1)

    resume_file = sys.argv[1]

    # Initialize parser and scorer
    print("Initializing parser and scorer...")
    parser = ResumeParser(enable_nlp=False)
    scorer = CandidateScorer()

    # Parse resume
    print(f"\nParsing resume: {resume_file}")
    resume = parser.parse(resume_file)

    if resume.parsing_errors:
        print(f"\nErrors parsing resume:")
        for error in resume.parsing_errors:
            print(f"  - {error}")
        sys.exit(1)

    # Define job requirements (example for a Python developer)
    job_requirements = JobRequirements(
        required_skills=[
            "Python",
            "Django",
            "PostgreSQL",
            "REST API",
            "Git",
        ],
        preferred_skills=[
            "Docker",
            "AWS",
            "React",
            "CI/CD",
            "TDD",
        ],
        min_years_experience=3,
        max_years_experience=8,
        required_education="Bachelor",
        job_title="Senior Python Developer",
        industry="Technology",
        keywords=["web development", "backend", "microservices"],
    )

    # Score candidate
    print("\nScoring candidate against job requirements...")
    print("\nJob Requirements:")
    print(f"  Title: {job_requirements.job_title}")
    print(f"  Required Skills: {', '.join(job_requirements.required_skills)}")
    print(f"  Preferred Skills: {', '.join(job_requirements.preferred_skills)}")
    print(f"  Experience: {job_requirements.min_years_experience}-{job_requirements.max_years_experience} years")

    result = scorer.score_candidate(resume, job_requirements)

    # Display results
    print("\n" + "=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)

    print(f"\nOverall Score: {result.overall_score:.1f}/100")
    print(f"Ranking: {result.ranking}")

    print("\n--- DIMENSION SCORES ---")
    for dimension, score in result.dimension_scores.items():
        print(f"  {dimension.replace('_', ' ').title()}: {score:.1f}/100")

    print("\n--- MATCH DETAILS ---")
    if result.match_details.get("matched_required_skills"):
        print(f"\nMatched Required Skills ({len(result.match_details['matched_required_skills'])}):")
        for skill in result.match_details["matched_required_skills"]:
            print(f"  ✓ {skill}")

    if result.match_details.get("missing_required_skills"):
        print(f"\nMissing Required Skills ({len(result.match_details['missing_required_skills'])}):")
        for skill in result.match_details["missing_required_skills"]:
            print(f"  ✗ {skill}")

    if result.match_details.get("total_experience_years"):
        print(f"\nTotal Experience: {result.match_details['total_experience_years']} years")

    if result.match_details.get("education_level"):
        print(f"Education Level: {result.match_details['education_level']}")

    print("\n--- RECOMMENDATIONS ---")
    if result.recommendations:
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  No specific recommendations - good match overall!")

    print("\n" + "=" * 60)

    # Summary
    if result.overall_score >= 75:
        print("\n✓ RECOMMEND for interview")
    elif result.overall_score >= 60:
        print("\n⚠ CONSIDER for interview")
    else:
        print("\n✗ NOT RECOMMENDED at this time")

    print()


if __name__ == "__main__":
    main()
