"""
Feature Engineering Demo

This example demonstrates the complete feature engineering pipeline:
1. Parse resumes
2. Extract features
3. Transform and normalize features
4. Store features for later use
5. Analyze feature statistics
"""

import sys
from pathlib import Path

# Add parent directory to path to import candiapp
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candiapp.parser import ResumeParser
from candiapp.features import (
    FeatureExtractor,
    FeatureTransformer,
    create_feature_pipeline,
)
from candiapp.feature_store import FeatureStore, FeatureIndex
from candiapp.models import ParsedData, Resume


def demo_basic_feature_extraction():
    """Demo 1: Basic feature extraction from a resume."""
    print("=" * 80)
    print("Demo 1: Basic Feature Extraction")
    print("=" * 80)

    # Create sample resume (in a real scenario, you would parse from a file)
    from candiapp.models import (
        ContactInfo,
        Experience,
        Education,
        Skill,
        ExperienceLevel,
        EducationLevel,
    )
    from datetime import datetime

    sample_data = ParsedData(
        full_name="Jane Smith",
        contact=ContactInfo(
            email="jane.smith@example.com",
            phone="+972-50-123-4567",
            linkedin="https://linkedin.com/in/janesmith",
        ),
        summary="Experienced data scientist with expertise in machine learning and AI",
        experiences=[
            Experience(
                company="AI Solutions Ltd",
                title="Senior Data Scientist",
                start_date=datetime(2020, 1, 1),
                current=True,
                responsibilities=[
                    "Lead ML model development",
                    "Mentor junior data scientists",
                    "Present insights to stakeholders",
                ],
            ),
        ],
        education=[
            Education(
                institution="Technion",
                degree="M.Sc. Computer Science",
                field_of_study="Machine Learning",
                level=EducationLevel.MASTER,
                gpa=3.9,
            ),
        ],
        skills=[
            Skill(name="Python", proficiency="Expert", years_of_experience=8),
            Skill(name="TensorFlow", proficiency="Advanced", years_of_experience=5),
            Skill(name="scikit-learn", proficiency="Expert", years_of_experience=6),
        ],
        technical_skills=["Python", "R", "SQL", "TensorFlow", "PyTorch", "AWS"],
        total_experience_years=7.5,
        experience_level=ExperienceLevel.SENIOR,
    )

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_features(sample_data)

    print(f"\n✓ Extracted {features.feature_count} features")
    print(f"\n📊 Feature Breakdown:")
    print(f"  - Numerical features: {len(features.numerical_features)}")
    print(f"  - Categorical features: {len(features.categorical_features)}")
    print(f"  - Text features: {len(features.text_features)}")
    print(f"  - List features: {len(features.list_features)}")

    print(f"\n🔢 Sample Numerical Features:")
    for name, value in list(features.numerical_features.items())[:10]:
        print(f"  - {name}: {value:.2f}")

    print(f"\n📝 Sample Categorical Features:")
    for name, value in list(features.categorical_features.items())[:5]:
        print(f"  - {name}: {value}")

    return features


def demo_feature_transformation():
    """Demo 2: Feature transformation and normalization."""
    print("\n" + "=" * 80)
    print("Demo 2: Feature Transformation and Normalization")
    print("=" * 80)

    from candiapp.models import (
        ParsedData,
        ContactInfo,
        Experience,
        ExperienceLevel,
    )
    from datetime import datetime

    # Create multiple sample resumes with different experience levels
    samples = []

    for i, (name, years) in enumerate([
        ("Junior Dev", 2.0),
        ("Mid-level Dev", 5.0),
        ("Senior Dev", 10.0),
    ]):
        data = ParsedData(
            full_name=name,
            contact=ContactInfo(email=f"{name.lower().replace(' ', '.')}@example.com"),
            total_experience_years=years,
            experiences=[
                Experience(
                    company="Tech Company",
                    title=name,
                    start_date=datetime(2024 - int(years), 1, 1),
                    current=True,
                )
            ],
        )
        samples.append(data)

    # Extract features
    extractor = FeatureExtractor()
    feature_vectors = [extractor.extract_features(sample) for sample in samples]

    print(f"\n✓ Created {len(feature_vectors)} feature vectors")

    # Show original values
    print(f"\n📊 Original Experience Years:")
    for i, fv in enumerate(feature_vectors):
        exp_years = fv.numerical_features.get("exp_total_years", 0)
        print(f"  - Resume {i+1}: {exp_years} years")

    # Transform with min-max normalization
    transformer = FeatureTransformer()
    normalized = transformer.fit_transform(feature_vectors, method="minmax")

    print(f"\n🔄 Normalized Experience Years (Min-Max [0,1]):")
    for i, fv in enumerate(normalized):
        exp_years = fv.numerical_features.get("exp_total_years", 0)
        print(f"  - Resume {i+1}: {exp_years:.3f}")

    # Transform with z-score normalization
    transformer2 = FeatureTransformer()
    standardized = transformer2.fit_transform(feature_vectors, method="zscore")

    print(f"\n📈 Standardized Experience Years (Z-Score):")
    for i, fv in enumerate(standardized):
        exp_years = fv.numerical_features.get("exp_total_years", 0)
        print(f"  - Resume {i+1}: {exp_years:.3f}")


def demo_feature_storage():
    """Demo 3: Storing and retrieving features."""
    print("\n" + "=" * 80)
    print("Demo 3: Feature Storage and Retrieval")
    print("=" * 80)

    import tempfile
    import shutil

    # Create temporary storage directory
    temp_dir = tempfile.mkdtemp()
    print(f"\n📁 Created temporary storage at: {temp_dir}")

    try:
        # Initialize feature store
        store = FeatureStore(storage_path=temp_dir, backend="json")

        # Create some sample features
        from candiapp.models import ParsedData, ContactInfo
        extractor = FeatureExtractor()

        resumes = {}
        for i in range(5):
            data = ParsedData(
                full_name=f"Candidate {i+1}",
                contact=ContactInfo(email=f"candidate{i+1}@example.com"),
            )
            features = extractor.extract_features(data)
            resume_id = f"resume_{i+1:03d}"
            resumes[resume_id] = features

        # Batch save
        print(f"\n💾 Saving {len(resumes)} feature vectors...")
        store.batch_save(resumes)

        # List all stored features
        all_ids = store.list_all()
        print(f"\n✓ Stored features for {len(all_ids)} resumes")
        print(f"  IDs: {', '.join(all_ids)}")

        # Load a specific feature vector
        loaded = store.load("resume_001")
        print(f"\n📂 Loaded features for resume_001:")
        print(f"  - Feature count: {loaded.feature_count}")

        # Get storage statistics
        stats = store.get_stats()
        print(f"\n📊 Storage Statistics:")
        print(f"  - Total features: {stats['total_features']}")
        print(f"  - Storage size: {stats['total_storage_mb']:.2f} MB")
        print(f"  - Backend: {stats['backend']}")

    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        print(f"\n🗑️  Cleaned up temporary storage")


def demo_feature_index():
    """Demo 4: Feature indexing and search."""
    print("\n" + "=" * 80)
    print("Demo 4: Feature Indexing and Search")
    print("=" * 80)

    from candiapp.models import ParsedData, ContactInfo, Skill

    # Create index
    index = FeatureIndex()
    extractor = FeatureExtractor()

    # Add some resumes with different skills
    samples = [
        ("Python Developer", ["Python", "Django", "PostgreSQL"]),
        ("JavaScript Developer", ["JavaScript", "React", "Node.js"]),
        ("Full Stack Developer", ["Python", "JavaScript", "React", "PostgreSQL"]),
        ("Data Scientist", ["Python", "TensorFlow", "Pandas"]),
    ]

    print(f"\n📇 Indexing {len(samples)} resumes...")

    for i, (title, skills) in enumerate(samples):
        data = ParsedData(
            full_name=title,
            contact=ContactInfo(email=f"{title.lower().replace(' ', '.')}@example.com"),
            technical_skills=skills,
        )
        features = extractor.extract_features(data)
        resume_id = f"resume_{i+1:03d}"
        index.add(resume_id, features)

    print(f"✓ Indexed {len(samples)} resumes")

    # Get all feature names
    all_features = index.get_all_feature_names()
    print(f"\n🔍 Total unique features: {len(all_features)}")

    # Get feature coverage
    coverage = index.get_feature_coverage()
    print(f"\n📊 Most common features:")
    sorted_coverage = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
    for feat_name, count in sorted_coverage[:10]:
        print(f"  - {feat_name}: {count} resumes")

    # Search for specific features
    required = ["skills_technical_count", "completeness_score"]
    matching = index.search_by_features(required)
    print(f"\n🔎 Resumes with features {required}:")
    print(f"  Found {len(matching)} matching resumes: {', '.join(matching)}")


def demo_complete_pipeline():
    """Demo 5: Complete end-to-end pipeline."""
    print("\n" + "=" * 80)
    print("Demo 5: Complete Feature Engineering Pipeline")
    print("=" * 80)

    from candiapp.models import (
        ParsedData,
        ContactInfo,
        Experience,
        Education,
        Skill,
        EducationLevel,
    )
    from datetime import datetime

    # Create multiple sample resumes
    samples = []

    # Sample 1: Experienced developer
    samples.append(
        ParsedData(
            full_name="Alice Johnson",
            contact=ContactInfo(email="alice@example.com", phone="+1-555-0001"),
            total_experience_years=8.0,
            experiences=[
                Experience(
                    company="Big Tech Corp",
                    title="Senior Engineer",
                    start_date=datetime(2016, 1, 1),
                    current=True,
                )
            ],
            education=[
                Education(
                    institution="MIT",
                    degree="M.Sc.",
                    field_of_study="Computer Science",
                    level=EducationLevel.MASTER,
                )
            ],
            skills=[
                Skill(name="Python", proficiency="Expert"),
                Skill(name="Java", proficiency="Advanced"),
            ],
        )
    )

    # Sample 2: Junior developer
    samples.append(
        ParsedData(
            full_name="Bob Smith",
            contact=ContactInfo(email="bob@example.com"),
            total_experience_years=2.0,
            experiences=[
                Experience(
                    company="Startup Inc",
                    title="Junior Developer",
                    start_date=datetime(2022, 1, 1),
                    current=True,
                )
            ],
            education=[
                Education(
                    institution="Local University",
                    degree="B.Sc.",
                    level=EducationLevel.BACHELOR,
                )
            ],
        )
    )

    # Run complete pipeline
    print(f"\n🚀 Running pipeline on {len(samples)} resumes...")

    feature_vectors, transformer = create_feature_pipeline(
        samples, transform=True, transform_method="minmax"
    )

    print(f"\n✓ Pipeline completed!")
    print(f"  - Extracted features: {len(feature_vectors)}")
    print(f"  - Transformer fitted: {transformer.is_fitted}")

    # Show results
    print(f"\n📊 Feature Comparison:")
    print(f"\n{'Feature':<30} {'Alice (Senior)':<20} {'Bob (Junior)':<20}")
    print("-" * 70)

    # Compare a few key features
    key_features = [
        "exp_total_years",
        "completeness_score",
        "edu_count",
        "skills_count",
    ]

    for feat in key_features:
        alice_val = feature_vectors[0].numerical_features.get(feat, 0)
        bob_val = feature_vectors[1].numerical_features.get(feat, 0)
        print(f"{feat:<30} {alice_val:<20.2f} {bob_val:<20.2f}")

    # Convert to numpy arrays for ML
    print(f"\n🔢 Converting to ML-ready arrays...")
    feature_names = sorted(feature_vectors[0].numerical_features.keys())
    arrays = [fv.to_array(feature_names) for fv in feature_vectors]

    print(f"✓ Created {len(arrays)} feature arrays")
    print(f"  - Shape: ({len(arrays)}, {len(feature_names)})")
    print(f"  - Ready for ML models!")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print(" 🎯 CandiApp Feature Engineering Demonstration")
    print("=" * 80)

    demo_basic_feature_extraction()
    demo_feature_transformation()
    demo_feature_storage()
    demo_feature_index()
    demo_complete_pipeline()

    print("\n" + "=" * 80)
    print("✨ All demos completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Parse real resumes using ResumeParser")
    print("  2. Extract features using FeatureExtractor")
    print("  3. Store features using FeatureStore")
    print("  4. Use features for candidate scoring and matching")
    print()


if __name__ == "__main__":
    main()
