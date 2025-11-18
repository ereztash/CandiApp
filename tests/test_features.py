"""
Tests for feature engineering module.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from candiapp.models import (
    ParsedData,
    Resume,
    Experience,
    Education,
    Skill,
    ContactInfo,
    ExperienceLevel,
    EducationLevel,
)
from candiapp.features import (
    FeatureExtractor,
    FeatureTransformer,
    FeatureVector,
    create_feature_pipeline,
)
from candiapp.feature_store import FeatureStore, FeatureIndex


@pytest.fixture
def sample_parsed_data():
    """Create a sample ParsedData object for testing."""
    contact = ContactInfo(
        email="test@example.com",
        phone="+1-234-567-8900",
        linkedin="https://linkedin.com/in/testuser",
        github="https://github.com/testuser",
    )

    experiences = [
        Experience(
            company="Tech Corp",
            title="Senior Software Engineer",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2023, 12, 31),
            current=False,
            location="Tel Aviv, Israel",
            description="Led development of microservices",
            responsibilities=["Architecture design", "Code reviews", "Mentoring"],
            achievements=["Reduced latency by 50%", "Improved test coverage to 90%"],
        ),
        Experience(
            company="Startup Inc",
            title="Software Engineer",
            start_date=datetime(2017, 6, 1),
            end_date=datetime(2019, 12, 31),
            current=False,
            location="New York, USA",
            description="Full-stack development",
            responsibilities=["Feature development", "Bug fixing"],
            achievements=["Launched 3 major features"],
        ),
    ]

    education = [
        Education(
            institution="Tel Aviv University",
            degree="B.Sc. Computer Science",
            field_of_study="Computer Science",
            start_date=datetime(2013, 9, 1),
            end_date=datetime(2017, 6, 1),
            gpa=3.8,
            level=EducationLevel.BACHELOR,
        ),
    ]

    skills = [
        Skill(name="Python", category="Programming", proficiency="Expert", years_of_experience=7),
        Skill(name="JavaScript", category="Programming", proficiency="Advanced", years_of_experience=5),
        Skill(name="React", category="Web", proficiency="Advanced", years_of_experience=4),
        Skill(name="AWS", category="Cloud", proficiency="Intermediate", years_of_experience=3),
    ]

    parsed_data = ParsedData(
        full_name="John Doe",
        first_name="John",
        last_name="Doe",
        contact=contact,
        summary="Experienced software engineer with 7 years of experience in full-stack development.",
        experiences=experiences,
        total_experience_years=6.5,
        experience_level=ExperienceLevel.SENIOR,
        education=education,
        highest_education=EducationLevel.BACHELOR,
        skills=skills,
        technical_skills=["Python", "JavaScript", "React", "AWS", "Docker", "Kubernetes"],
        soft_skills=["Leadership", "Communication", "Problem Solving"],
        languages=[],
        certifications=[],
        projects=[],
    )

    return parsed_data


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_basic_feature_extraction(self, sample_parsed_data):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        assert isinstance(features, FeatureVector)
        assert features.feature_count > 0
        assert len(features.numerical_features) > 0

    def test_experience_features(self, sample_parsed_data):
        """Test experience feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Check basic experience features
        assert features.numerical_features["exp_count"] == 2.0
        assert features.numerical_features["exp_total_years"] == 6.5

        # Check employment status
        assert features.numerical_features["exp_currently_employed"] == 0.0

        # Check company diversity
        assert features.numerical_features["exp_company_diversity"] == 2.0

        # Check responsibilities and achievements
        assert features.numerical_features["exp_total_responsibilities"] == 5.0
        assert features.numerical_features["exp_total_achievements"] == 3.0

    def test_education_features(self, sample_parsed_data):
        """Test education feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Check education count
        assert features.numerical_features["edu_count"] == 1.0

        # Check GPA
        assert features.numerical_features["edu_max_gpa"] == 3.8

        # Check STEM field detection
        assert features.numerical_features["edu_has_stem"] == 1.0

        # Check categorical features
        assert features.categorical_features["edu_highest_level"] == "bachelor"

    def test_skills_features(self, sample_parsed_data):
        """Test skills feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Check skill counts
        assert features.numerical_features["skills_count"] == 4.0
        assert features.numerical_features["skills_technical_count"] == 6.0
        assert features.numerical_features["skills_soft_count"] == 3.0

        # Check years of experience
        assert features.numerical_features["skills_max_years"] == 7.0

        # Check skill lists
        assert "Python" in features.list_features["skills_all"]
        assert "Python" in features.list_features["skills_technical"]

    def test_completeness_features(self, sample_parsed_data):
        """Test completeness feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Completeness score should be high for a well-filled resume
        assert features.numerical_features["completeness_score"] > 70.0

    def test_text_features(self, sample_parsed_data):
        """Test text feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Check contact information
        assert features.numerical_features["text_has_email"] == 1.0
        assert features.numerical_features["text_has_phone"] == 1.0
        assert features.numerical_features["text_has_linkedin"] == 1.0
        assert features.numerical_features["text_has_github"] == 1.0

        # Check summary
        assert features.numerical_features["text_summary_length"] > 0

    def test_temporal_features(self, sample_parsed_data):
        """Test temporal feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Check most recent experience year
        assert features.numerical_features["temporal_most_recent_exp_year"] == 2023.0

        # Years since last experience should be calculated
        assert "temporal_years_since_last_exp" in features.numerical_features

    def test_diversity_features(self, sample_parsed_data):
        """Test diversity feature extraction."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Check certification count (should be 0 for sample data)
        assert features.numerical_features["diversity_certification_count"] == 0.0

    def test_seniority_progression(self, sample_parsed_data):
        """Test seniority progression analysis."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Should detect progression from Software Engineer to Senior Software Engineer
        assert features.numerical_features["exp_seniority_trend"] > 0


class TestFeatureTransformer:
    """Tests for FeatureTransformer class."""

    def test_minmax_normalization(self, sample_parsed_data):
        """Test min-max normalization."""
        # Create multiple feature vectors
        extractor = FeatureExtractor()
        fv1 = extractor.extract_features(sample_parsed_data)

        # Create a second sample with different values
        sample2 = sample_parsed_data
        sample2.total_experience_years = 10.0
        fv2 = extractor.extract_features(sample2)

        # Transform
        transformer = FeatureTransformer()
        transformed = transformer.fit_transform([fv1, fv2], method="minmax")

        assert len(transformed) == 2
        # Values should be normalized to [0, 1]
        for fv in transformed:
            for value in fv.numerical_features.values():
                assert 0.0 <= value <= 1.0 or abs(value - 0.5) < 0.01  # Allow 0.5 for constant features

    def test_zscore_normalization(self, sample_parsed_data):
        """Test z-score normalization."""
        extractor = FeatureExtractor()
        fv = extractor.extract_features(sample_parsed_data)

        transformer = FeatureTransformer()
        transformed = transformer.fit_transform([fv], method="zscore")

        assert len(transformed) == 1

    def test_no_transformation(self, sample_parsed_data):
        """Test that 'none' method doesn't transform."""
        extractor = FeatureExtractor()
        fv = extractor.extract_features(sample_parsed_data)

        original_values = fv.numerical_features.copy()

        transformer = FeatureTransformer()
        transformed = transformer.transform(fv, method="none")

        # Values should remain unchanged
        assert transformed.numerical_features == original_values


class TestFeatureVector:
    """Tests for FeatureVector class."""

    def test_to_array(self):
        """Test conversion to numpy array."""
        fv = FeatureVector(
            numerical_features={"feat1": 1.0, "feat2": 2.0, "feat3": 3.0}
        )

        arr = fv.to_array(["feat1", "feat2", "feat3"])
        assert len(arr) == 3
        assert arr[0] == 1.0
        assert arr[1] == 2.0
        assert arr[2] == 3.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        fv = FeatureVector(
            numerical_features={"feat1": 1.0},
            categorical_features={"cat1": "value1"},
            text_features={"text1": "some text"},
            list_features={"list1": ["item1", "item2"]},
        )

        d = fv.to_dict()
        assert "numerical" in d
        assert "categorical" in d
        assert "text" in d
        assert "list" in d
        assert "metadata" in d


class TestFeaturePipeline:
    """Tests for feature engineering pipeline."""

    def test_basic_pipeline(self, sample_parsed_data):
        """Test basic feature pipeline."""
        feature_vectors, transformer = create_feature_pipeline(
            [sample_parsed_data],
            transform=True,
            transform_method="minmax"
        )

        assert len(feature_vectors) == 1
        assert transformer is not None
        assert transformer.is_fitted

    def test_pipeline_without_transform(self, sample_parsed_data):
        """Test pipeline without transformation."""
        feature_vectors, transformer = create_feature_pipeline(
            [sample_parsed_data],
            transform=False
        )

        assert len(feature_vectors) == 1
        assert transformer is None


class TestFeatureStore:
    """Tests for FeatureStore class."""

    @pytest.fixture
    def temp_store_path(self):
        """Create a temporary directory for feature store."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_save_and_load_json(self, sample_parsed_data, temp_store_path):
        """Test saving and loading features with JSON backend."""
        store = FeatureStore(storage_path=temp_store_path, backend="json")

        # Extract and save features
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        resume_id = "test_resume_001"
        store.save(resume_id, features)

        # Load features
        loaded_features = store.load(resume_id)

        assert loaded_features is not None
        assert loaded_features.feature_count == features.feature_count
        assert len(loaded_features.numerical_features) == len(features.numerical_features)

    def test_save_and_load_pickle(self, sample_parsed_data, temp_store_path):
        """Test saving and loading features with pickle backend."""
        store = FeatureStore(storage_path=temp_store_path, backend="pickle")

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        resume_id = "test_resume_002"
        store.save(resume_id, features)

        loaded_features = store.load(resume_id)

        assert loaded_features is not None
        assert loaded_features.feature_count == features.feature_count

    def test_delete(self, sample_parsed_data, temp_store_path):
        """Test deleting features."""
        store = FeatureStore(storage_path=temp_store_path, backend="json")

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        resume_id = "test_resume_003"
        store.save(resume_id, features)

        # Delete
        store.delete(resume_id)

        # Try to load (should return None)
        loaded = store.load(resume_id)
        assert loaded is None

    def test_list_all(self, sample_parsed_data, temp_store_path):
        """Test listing all features."""
        store = FeatureStore(storage_path=temp_store_path, backend="json")

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Save multiple features
        store.save("resume_001", features)
        store.save("resume_002", features)
        store.save("resume_003", features)

        all_ids = store.list_all()
        assert len(all_ids) == 3
        assert "resume_001" in all_ids
        assert "resume_002" in all_ids
        assert "resume_003" in all_ids

    def test_batch_operations(self, sample_parsed_data, temp_store_path):
        """Test batch save and load."""
        store = FeatureStore(storage_path=temp_store_path, backend="json")

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        # Batch save
        batch = {
            "resume_001": features,
            "resume_002": features,
            "resume_003": features,
        }
        store.batch_save(batch)

        # Batch load
        loaded = store.batch_load(["resume_001", "resume_002", "resume_003"])
        assert len(loaded) == 3

    def test_cache(self, sample_parsed_data, temp_store_path):
        """Test caching functionality."""
        store = FeatureStore(storage_path=temp_store_path, backend="json")

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        resume_id = "test_resume_cache"
        store.save(resume_id, features)

        # First load (from disk)
        features1 = store.load(resume_id)
        assert resume_id in store.cache

        # Second load (from cache)
        features2 = store.load(resume_id)
        assert features1 is features2  # Should be same object from cache

        # Clear cache
        store.clear_cache()
        assert len(store.cache) == 0

    def test_stats(self, sample_parsed_data, temp_store_path):
        """Test store statistics."""
        store = FeatureStore(storage_path=temp_store_path, backend="json")

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        store.save("resume_001", features)
        store.save("resume_002", features)

        stats = store.get_stats()
        assert stats["total_features"] == 2
        assert stats["backend"] == "json"
        assert stats["total_storage_bytes"] > 0


class TestFeatureIndex:
    """Tests for FeatureIndex class."""

    def test_add_and_search(self, sample_parsed_data):
        """Test adding features to index and searching."""
        index = FeatureIndex()

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        index.add("resume_001", features)

        # Search for resumes with specific features
        results = index.search_by_features(["exp_count", "edu_count"])
        assert "resume_001" in results

    def test_get_all_feature_names(self, sample_parsed_data):
        """Test getting all feature names."""
        index = FeatureIndex()

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        index.add("resume_001", features)

        all_features = index.get_all_feature_names()
        assert len(all_features) > 0
        assert "exp_count" in all_features

    def test_feature_coverage(self, sample_parsed_data):
        """Test feature coverage statistics."""
        index = FeatureIndex()

        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_parsed_data)

        index.add("resume_001", features)
        index.add("resume_002", features)

        coverage = index.get_feature_coverage()
        assert coverage["exp_count"] == 2  # Present in both resumes
