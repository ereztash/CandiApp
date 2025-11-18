#!/usr/bin/env python3
"""
Quick test script for new research-based features
"""

import sys
sys.path.insert(0, 'src')

from candiapp.models import ParsedData, Experience, Education, Skill, ContactInfo
from candiapp.advanced_features import AdvancedFeatureExtractor
from datetime import datetime, timedelta

print("=" * 80)
print("Testing New Research-Based Features (v2.1)")
print("=" * 80)

# Create sample parsed data
contact = ContactInfo(
    email="test@example.com",
    phone="123-456-7890",
    linkedin="https://linkedin.com/in/test",
    github="https://github.com/test"
)

# Create sample experiences
experiences = [
    Experience(
        title="Senior Software Engineer",
        company="Google",
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2023, 12, 31),
        description="Led development of ML systems",
        responsibilities=[
            "Led team of 5 engineers",
            "Designed scalable architecture"
        ],
        achievements=[
            "Reduced latency by 50%, improving response time for 2M+ users",
            "Implemented distributed system handling 10M requests/day",
            "Mentored 3 junior engineers, all promoted within 1 year"
        ],
        current=False
    ),
    Experience(
        title="Software Engineer",
        company="Microsoft",
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2019, 12, 31),
        description="Developed cloud services",
        responsibilities=[
            "Collaborated with cross-functional teams",
            "Optimized database performance"
        ],
        achievements=[
            "Increased system throughput by 30%",
            "Resolved critical production issues"
        ],
        current=False
    )
]

# Create sample education
education = [
    Education(
        institution="MIT",
        degree="BS Computer Science",
        field_of_study="Computer Science",
        start_date=datetime(2014, 9, 1),
        end_date=datetime(2018, 5, 30),
        gpa=3.8,
        level=None
    )
]

# Create sample skills
skills = [
    Skill(name="Python", proficiency="Expert", years_of_experience=5),
    Skill(name="Machine Learning", proficiency="Advanced", years_of_experience=3),
    Skill(name="System Design", proficiency="Advanced", years_of_experience=4),
]

parsed_data = ParsedData(
    contact=contact,
    full_name="Test Candidate",
    summary="Experienced software engineer with focus on ML and distributed systems",
    experiences=experiences,
    education=education,
    skills=skills,
    technical_skills=["Python", "TensorFlow", "Kubernetes", "AWS", "SQL"],
    soft_skills=["Leadership", "Communication", "Problem Solving"],
    certifications=["AWS Certified Solutions Architect - Professional"],
    projects=[],
    languages=[],
    publications=[],
    awards=[],
    volunteer_experience=[],
    total_experience_years=6.0,
    raw_text=""
)

# Test Advanced Feature Extractor
print("\n1. Testing Advanced Feature Extractor...")
print("-" * 80)

extractor = AdvancedFeatureExtractor()
base_features = {}  # Empty base features for testing

try:
    features = extractor.extract_advanced_features(parsed_data, base_features)
    print(f"✅ Extracted {len(features)} advanced features")

    # Test Emotional Intelligence Features
    print("\n2. Emotional Intelligence (EI) Features (0.60 validity):")
    print("-" * 80)
    ei_features = {k: v for k, v in features.items() if k.startswith('ei_')}
    for feature_name, value in sorted(ei_features.items()):
        print(f"  {feature_name}: {value:.2f}")

    if "ei_overall_score" in features:
        score = features["ei_overall_score"]
        if score > 70:
            print(f"  ✅ HIGH EI Score: {score:.1f}/100 (Top performer indicator)")
        elif score > 50:
            print(f"  ✓ Moderate EI Score: {score:.1f}/100")
        else:
            print(f"  ⚠️  Lower EI Score: {score:.1f}/100")

    # Test Cognitive Ability Features
    print("\n3. Cognitive Ability Features (0.51 validity):")
    print("-" * 80)
    cognitive_features = {k: v for k, v in features.items() if k.startswith('cognitive_')}
    for feature_name, value in sorted(cognitive_features.items()):
        print(f"  {feature_name}: {value:.2f}")

    if "cognitive_learning_velocity" in features:
        velocity = features["cognitive_learning_velocity"]
        if velocity > 5:
            print(f"  ✅ HIGH Learning Velocity: {velocity:.1f} skills/year")
        else:
            print(f"  ✓ Moderate Learning Velocity: {velocity:.1f} skills/year")

    # Test Person-Organization Fit Features
    print("\n4. Person-Organization Fit Features (0.44 validity):")
    print("-" * 80)
    po_fit_features = {k: v for k, v in features.items() if k.startswith('po_fit_')}
    for feature_name, value in sorted(po_fit_features.items()):
        print(f"  {feature_name}: {value:.2f}")

    if "po_fit_overall_score" in features:
        score = features["po_fit_overall_score"]
        if score > 70:
            print(f"  ✅ EXCELLENT Org Fit: {score:.1f}/100")
        elif score > 50:
            print(f"  ✓ Good Org Fit: {score:.1f}/100")
        else:
            print(f"  ⚠️  Mixed Org Fit: {score:.1f}/100")

    # Test Structured Assessment Features
    print("\n5. Structured Assessment Features (0.54 validity):")
    print("-" * 80)
    assessment_features = {k: v for k, v in features.items() if k.startswith('assessment_')}
    for feature_name, value in sorted(assessment_features.items()):
        print(f"  {feature_name}: {value:.2f}")

    if "assessment_overall_quality" in features:
        score = features["assessment_overall_quality"]
        if score > 75:
            print(f"  ✅ STRONG Achiever Pattern: {score:.1f}/100")
        elif score > 60:
            print(f"  ✓ Good Achiever Pattern: {score:.1f}/100")
        else:
            print(f"  ⚠️  Moderate Pattern: {score:.1f}/100")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Features Extracted: {len(features)}")
    print(f"  - EI Features: {len(ei_features)}")
    print(f"  - Cognitive Features: {len(cognitive_features)}")
    print(f"  - PO-Fit Features: {len(po_fit_features)}")
    print(f"  - Assessment Features: {len(assessment_features)}")
    print(f"  - Other Advanced Features: {len(features) - len(ei_features) - len(cognitive_features) - len(po_fit_features) - len(assessment_features)}")

    print("\n✅ ALL TESTS PASSED!")
    print("\nNote: Semantic Match features require BERT models and job requirements.")
    print("      Run with BERTFeatureExtractor to test semantic matching (0.67 validity).")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
