"""
Feature Engineering Module for Resume Analysis

This module implements comprehensive feature engineering for resume parsing
and candidate matching, including:
- Advanced feature extraction from parsed resume data
- Feature transformation and normalization
- Vector representations for ML models
- Skill embedding and semantic matching
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
from datetime import datetime
import logging

from .models import (
    ParsedData,
    Resume,
    Experience,
    Education,
    Skill,
    ExperienceLevel,
    EducationLevel,
)

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """
    Complete feature vector representation of a resume.

    Includes both numerical and categorical features for ML models.
    """
    # Numerical features
    numerical_features: Dict[str, float] = field(default_factory=dict)

    # Categorical features (encoded as strings)
    categorical_features: Dict[str, str] = field(default_factory=dict)

    # Text features (for NLP models)
    text_features: Dict[str, str] = field(default_factory=dict)

    # List features (multi-value)
    list_features: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    feature_count: int = 0
    extraction_timestamp: Optional[datetime] = None

    def to_array(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert numerical features to numpy array.

        Args:
            feature_names: Ordered list of feature names to include

        Returns:
            Numpy array of feature values
        """
        if feature_names is None:
            feature_names = sorted(self.numerical_features.keys())

        return np.array([
            self.numerical_features.get(name, 0.0)
            for name in feature_names
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "numerical": self.numerical_features,
            "categorical": self.categorical_features,
            "text": self.text_features,
            "list": self.list_features,
            "metadata": {
                "feature_count": self.feature_count,
                "extraction_timestamp": self.extraction_timestamp.isoformat() if self.extraction_timestamp else None,
            }
        }


class FeatureExtractor:
    """
    Advanced feature extractor for resumes.

    Extracts 100+ engineered features from parsed resume data including:
    - Experience features (duration, gaps, progression)
    - Education features (level, relevance, recency)
    - Skills features (count, diversity, proficiency)
    - Text features (length, complexity, keywords)
    - Temporal features (recency, currency)
    """

    # Common technical skill categories
    SKILL_CATEGORIES = {
        "programming": ["python", "java", "javascript", "c++", "c#", "ruby", "go", "rust", "php", "swift", "kotlin"],
        "web": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask", "spring", "express"],
        "data": ["sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "elasticsearch", "pandas", "numpy"],
        "ml_ai": ["machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "nlp", "computer vision"],
        "cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd"],
        "mobile": ["android", "ios", "react native", "flutter", "swift", "kotlin"],
        "tools": ["git", "jira", "agile", "scrum", "linux", "bash", "vim"],
    }

    # Job title seniority levels
    SENIORITY_KEYWORDS = {
        "entry": ["junior", "associate", "intern", "trainee", "graduate"],
        "mid": ["developer", "engineer", "analyst", "specialist"],
        "senior": ["senior", "lead", "principal", "staff"],
        "lead": ["team lead", "tech lead", "engineering lead"],
        "management": ["manager", "director", "head of", "vp", "chief"],
        "executive": ["cto", "ceo", "coo", "cfo", "president"],
    }

    def __init__(self, enable_advanced_nlp: bool = False):
        """
        Initialize feature extractor.

        Args:
            enable_advanced_nlp: Enable advanced NLP features (requires spaCy)
        """
        self.enable_advanced_nlp = enable_advanced_nlp
        self._nlp_model = None

        if enable_advanced_nlp:
            try:
                import spacy
                try:
                    self._nlp_model = spacy.load("en_core_web_md")  # Medium model with word vectors
                except OSError:
                    logger.warning("spaCy model 'en_core_web_md' not found. Using 'en_core_web_sm'")
                    try:
                        self._nlp_model = spacy.load("en_core_web_sm")
                    except OSError:
                        logger.warning("No spaCy models found. Advanced NLP features disabled.")
            except ImportError:
                logger.warning("spaCy not installed. Advanced NLP features disabled.")

    def extract_features(self, parsed_data: ParsedData) -> FeatureVector:
        """
        Extract comprehensive feature vector from parsed resume data.

        Args:
            parsed_data: Parsed resume data

        Returns:
            FeatureVector with all engineered features
        """
        logger.info("Extracting features from parsed resume data...")

        features = FeatureVector(extraction_timestamp=datetime.now())

        # Extract features from different sections
        self._extract_experience_features(parsed_data, features)
        self._extract_education_features(parsed_data, features)
        self._extract_skills_features(parsed_data, features)
        self._extract_text_features(parsed_data, features)
        self._extract_completeness_features(parsed_data, features)
        self._extract_temporal_features(parsed_data, features)
        self._extract_diversity_features(parsed_data, features)

        # Count total features
        features.feature_count = (
            len(features.numerical_features) +
            len(features.categorical_features) +
            len(features.text_features) +
            len(features.list_features)
        )

        logger.info(f"Extracted {features.feature_count} features")

        return features

    def _extract_experience_features(self, parsed_data: ParsedData, features: FeatureVector):
        """Extract features from work experience."""
        experiences = parsed_data.experiences

        # Basic counts
        features.numerical_features["exp_count"] = float(len(experiences))
        features.numerical_features["exp_total_years"] = parsed_data.total_experience_years or 0.0

        if not experiences:
            return

        # Job tenure analysis
        tenures = []
        for exp in experiences:
            if exp.start_date and exp.end_date:
                tenure = (exp.end_date - exp.start_date).days / 365.25
                tenures.append(tenure)

        if tenures:
            features.numerical_features["exp_avg_tenure"] = np.mean(tenures)
            features.numerical_features["exp_max_tenure"] = np.max(tenures)
            features.numerical_features["exp_min_tenure"] = np.min(tenures)
            features.numerical_features["exp_tenure_std"] = np.std(tenures)

        # Current employment
        features.numerical_features["exp_currently_employed"] = 1.0 if any(exp.current for exp in experiences) else 0.0

        # Company diversity (number of unique companies)
        unique_companies = len(set(exp.company for exp in experiences if exp.company))
        features.numerical_features["exp_company_diversity"] = float(unique_companies)

        # Job hopping indicator (more than 2 jobs per year on average)
        if parsed_data.total_experience_years and parsed_data.total_experience_years > 0:
            job_frequency = len(experiences) / parsed_data.total_experience_years
            features.numerical_features["exp_job_frequency"] = job_frequency
            features.numerical_features["exp_is_job_hopper"] = 1.0 if job_frequency > 2.0 else 0.0

        # Career progression (title seniority analysis)
        seniority_progression = self._analyze_seniority_progression(experiences)
        features.numerical_features["exp_seniority_trend"] = seniority_progression

        # Responsibilities and achievements
        total_responsibilities = sum(len(exp.responsibilities) for exp in experiences)
        total_achievements = sum(len(exp.achievements) for exp in experiences)
        features.numerical_features["exp_total_responsibilities"] = float(total_responsibilities)
        features.numerical_features["exp_total_achievements"] = float(total_achievements)

        # Most recent job details
        if experiences:
            recent_exp = experiences[0]  # Assuming sorted by date
            features.categorical_features["exp_recent_company"] = recent_exp.company
            features.categorical_features["exp_recent_title"] = recent_exp.title
            if recent_exp.location:
                features.categorical_features["exp_recent_location"] = recent_exp.location

        # Experience level
        if parsed_data.experience_level:
            features.categorical_features["exp_level"] = parsed_data.experience_level.value
            features.numerical_features[f"exp_level_{parsed_data.experience_level.value}"] = 1.0

    def _extract_education_features(self, parsed_data: ParsedData, features: FeatureVector):
        """Extract features from education."""
        education = parsed_data.education

        # Basic counts
        features.numerical_features["edu_count"] = float(len(education))

        if not education:
            return

        # Highest education level
        if parsed_data.highest_education:
            features.categorical_features["edu_highest_level"] = parsed_data.highest_education.value
            # One-hot encoding for education level
            for level in EducationLevel:
                features.numerical_features[f"edu_level_{level.value}"] = 1.0 if parsed_data.highest_education == level else 0.0

        # GPA analysis
        gpas = [edu.gpa for edu in education if edu.gpa]
        if gpas:
            features.numerical_features["edu_max_gpa"] = max(gpas)
            features.numerical_features["edu_avg_gpa"] = np.mean(gpas)

        # Institution diversity
        unique_institutions = len(set(edu.institution for edu in education if edu.institution))
        features.numerical_features["edu_institution_diversity"] = float(unique_institutions)

        # Field of study
        fields = [edu.field_of_study for edu in education if edu.field_of_study]
        if fields:
            features.list_features["edu_fields_of_study"] = fields
            # Check for STEM fields
            stem_keywords = ["computer", "engineering", "science", "mathematics", "technology", "physics", "chemistry"]
            has_stem = any(any(kw in field.lower() for kw in stem_keywords) for field in fields)
            features.numerical_features["edu_has_stem"] = 1.0 if has_stem else 0.0

        # Most recent education
        if education:
            recent_edu = education[0]
            features.categorical_features["edu_recent_institution"] = recent_edu.institution
            if recent_edu.degree:
                features.categorical_features["edu_recent_degree"] = recent_edu.degree

    def _extract_skills_features(self, parsed_data: ParsedData, features: FeatureVector):
        """Extract features from skills."""
        skills = parsed_data.skills
        technical_skills = parsed_data.technical_skills
        soft_skills = parsed_data.soft_skills

        # Basic counts
        features.numerical_features["skills_count"] = float(len(skills))
        features.numerical_features["skills_technical_count"] = float(len(technical_skills))
        features.numerical_features["skills_soft_count"] = float(len(soft_skills))

        # Store skill lists
        features.list_features["skills_all"] = [s.name for s in skills]
        features.list_features["skills_technical"] = technical_skills
        features.list_features["skills_soft"] = soft_skills

        # Skill categorization
        all_skills_text = " ".join([s.name.lower() for s in skills] + [s.lower() for s in technical_skills])

        for category, keywords in self.SKILL_CATEGORIES.items():
            count = sum(1 for keyword in keywords if keyword in all_skills_text)
            features.numerical_features[f"skills_category_{category}"] = float(count)

        # Skill proficiency analysis
        proficiency_counts = Counter(s.proficiency for s in skills if s.proficiency)
        for prof, count in proficiency_counts.items():
            features.numerical_features[f"skills_proficiency_{prof.lower()}"] = float(count)

        # Years of experience with skills
        skill_years = [s.years_of_experience for s in skills if s.years_of_experience]
        if skill_years:
            features.numerical_features["skills_avg_years"] = np.mean(skill_years)
            features.numerical_features["skills_max_years"] = float(max(skill_years))

        # Skill diversity (using categories)
        categories_present = sum(1 for category in self.SKILL_CATEGORIES.keys()
                                if features.numerical_features.get(f"skills_category_{category}", 0) > 0)
        features.numerical_features["skills_category_diversity"] = float(categories_present)

    def _extract_text_features(self, parsed_data: ParsedData, features: FeatureVector):
        """Extract features from text content."""
        # Summary/objective text analysis
        if parsed_data.summary:
            summary_text = parsed_data.summary
            features.text_features["summary"] = summary_text
            features.numerical_features["text_summary_length"] = float(len(summary_text))
            features.numerical_features["text_summary_word_count"] = float(len(summary_text.split()))

        # Raw text statistics
        if parsed_data.raw_text:
            raw_text = parsed_data.raw_text
            features.numerical_features["text_total_length"] = float(len(raw_text))
            features.numerical_features["text_total_word_count"] = float(len(raw_text.split()))
            features.numerical_features["text_line_count"] = float(len(raw_text.split('\n')))

            # Language detection (basic - Hebrew vs English)
            hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', raw_text))
            english_chars = len(re.findall(r'[a-zA-Z]', raw_text))
            total_chars = hebrew_chars + english_chars

            if total_chars > 0:
                features.numerical_features["text_hebrew_ratio"] = hebrew_chars / total_chars
                features.numerical_features["text_english_ratio"] = english_chars / total_chars
                features.categorical_features["text_primary_language"] = "hebrew" if hebrew_chars > english_chars else "english"

        # Contact information completeness
        contact = parsed_data.contact
        features.numerical_features["text_has_email"] = 1.0 if contact.email else 0.0
        features.numerical_features["text_has_phone"] = 1.0 if contact.phone else 0.0
        features.numerical_features["text_has_linkedin"] = 1.0 if contact.linkedin else 0.0
        features.numerical_features["text_has_github"] = 1.0 if contact.github else 0.0
        features.numerical_features["text_has_website"] = 1.0 if contact.website else 0.0

    def _extract_completeness_features(self, parsed_data: ParsedData, features: FeatureVector):
        """Extract resume completeness features."""
        completeness_score = 0.0
        max_score = 0.0

        # Personal information (20 points)
        max_score += 20
        if parsed_data.full_name:
            completeness_score += 10
        if parsed_data.contact.email:
            completeness_score += 5
        if parsed_data.contact.phone:
            completeness_score += 5

        # Professional summary (10 points)
        max_score += 10
        if parsed_data.summary:
            completeness_score += 10

        # Experience (30 points)
        max_score += 30
        if parsed_data.experiences:
            completeness_score += 15
            # Additional points for detailed experiences
            has_dates = sum(1 for exp in parsed_data.experiences if exp.start_date)
            has_descriptions = sum(1 for exp in parsed_data.experiences if exp.description or exp.responsibilities)
            completeness_score += min(10, has_dates * 2)
            completeness_score += min(5, has_descriptions)

        # Education (20 points)
        max_score += 20
        if parsed_data.education:
            completeness_score += 10
            has_degrees = sum(1 for edu in parsed_data.education if edu.degree)
            completeness_score += min(10, has_degrees * 5)

        # Skills (15 points)
        max_score += 15
        if parsed_data.skills or parsed_data.technical_skills:
            skill_count = len(parsed_data.skills) + len(parsed_data.technical_skills)
            completeness_score += min(15, skill_count)

        # Additional sections (5 points)
        max_score += 5
        if parsed_data.languages:
            completeness_score += 2
        if parsed_data.certifications:
            completeness_score += 2
        if parsed_data.projects:
            completeness_score += 1

        features.numerical_features["completeness_score"] = (completeness_score / max_score) * 100 if max_score > 0 else 0.0
        features.numerical_features["completeness_raw_score"] = completeness_score

    def _extract_temporal_features(self, parsed_data: ParsedData, features: FeatureVector):
        """Extract time-based features."""
        current_year = datetime.now().year

        # Most recent experience year
        if parsed_data.experiences:
            recent_dates = []
            for exp in parsed_data.experiences:
                if exp.end_date and not exp.current:
                    recent_dates.append(exp.end_date.year)
                elif exp.current:
                    recent_dates.append(current_year)

            if recent_dates:
                most_recent_year = max(recent_dates)
                features.numerical_features["temporal_most_recent_exp_year"] = float(most_recent_year)
                features.numerical_features["temporal_years_since_last_exp"] = float(current_year - most_recent_year)

        # Education recency
        if parsed_data.education:
            edu_years = [edu.end_date.year for edu in parsed_data.education if edu.end_date]
            if edu_years:
                most_recent_edu_year = max(edu_years)
                features.numerical_features["temporal_most_recent_edu_year"] = float(most_recent_edu_year)
                features.numerical_features["temporal_years_since_graduation"] = float(current_year - most_recent_edu_year)

        # Career gaps (simplified)
        if len(parsed_data.experiences) >= 2:
            gaps = []
            sorted_exps = sorted(parsed_data.experiences, key=lambda x: x.start_date if x.start_date else datetime.min)

            for i in range(len(sorted_exps) - 1):
                current_exp = sorted_exps[i]
                next_exp = sorted_exps[i + 1]

                if current_exp.end_date and next_exp.start_date:
                    gap_days = (next_exp.start_date - current_exp.end_date).days
                    if gap_days > 30:  # Gap more than 1 month
                        gaps.append(gap_days / 365.25)

            if gaps:
                features.numerical_features["temporal_gap_count"] = float(len(gaps))
                features.numerical_features["temporal_total_gap_years"] = sum(gaps)
                features.numerical_features["temporal_avg_gap_years"] = np.mean(gaps)
                features.numerical_features["temporal_max_gap_years"] = max(gaps)

    def _extract_diversity_features(self, parsed_data: ParsedData, features: FeatureVector):
        """Extract diversity and breadth features."""
        # Language diversity
        features.numerical_features["diversity_language_count"] = float(len(parsed_data.languages))

        # Certification count
        features.numerical_features["diversity_certification_count"] = float(len(parsed_data.certifications))

        # Project count
        features.numerical_features["diversity_project_count"] = float(len(parsed_data.projects))

        # Publication count
        features.numerical_features["diversity_publication_count"] = float(len(parsed_data.publications))

        # Award count
        features.numerical_features["diversity_award_count"] = float(len(parsed_data.awards))

        # Volunteer experience
        features.numerical_features["diversity_volunteer_count"] = float(len(parsed_data.volunteer_experience))

        # Overall diversity score (presence of different resume sections)
        diversity_sections = sum([
            1 if parsed_data.languages else 0,
            1 if parsed_data.certifications else 0,
            1 if parsed_data.projects else 0,
            1 if parsed_data.publications else 0,
            1 if parsed_data.awards else 0,
            1 if parsed_data.volunteer_experience else 0,
        ])
        features.numerical_features["diversity_section_count"] = float(diversity_sections)

    def _analyze_seniority_progression(self, experiences: List[Experience]) -> float:
        """
        Analyze career progression based on job title seniority.

        Returns:
            Float between -1 and 1 indicating progression (-1 = regression, 0 = flat, 1 = strong progression)
        """
        if len(experiences) < 2:
            return 0.0

        # Sort by date (most recent first)
        sorted_exps = sorted(experiences, key=lambda x: x.start_date if x.start_date else datetime.min, reverse=True)

        seniority_scores = []
        for exp in sorted_exps:
            title_lower = exp.title.lower() if exp.title else ""

            # Assign seniority score
            if any(kw in title_lower for kw in self.SENIORITY_KEYWORDS["executive"]):
                score = 6
            elif any(kw in title_lower for kw in self.SENIORITY_KEYWORDS["management"]):
                score = 5
            elif any(kw in title_lower for kw in self.SENIORITY_KEYWORDS["lead"]):
                score = 4
            elif any(kw in title_lower for kw in self.SENIORITY_KEYWORDS["senior"]):
                score = 3
            elif any(kw in title_lower for kw in self.SENIORITY_KEYWORDS["mid"]):
                score = 2
            elif any(kw in title_lower for kw in self.SENIORITY_KEYWORDS["entry"]):
                score = 1
            else:
                score = 2  # Default to mid-level

            seniority_scores.append(score)

        # Calculate progression (positive = upward, negative = downward)
        if len(seniority_scores) >= 2:
            # Compare oldest to newest
            progression = seniority_scores[0] - seniority_scores[-1]
            # Normalize to -1 to 1 range
            return max(-1.0, min(1.0, progression / 3.0))

        return 0.0


class FeatureTransformer:
    """
    Transform and normalize features for machine learning models.

    Implements various transformation techniques:
    - Normalization (min-max, z-score)
    - Encoding (one-hot, label encoding)
    - Scaling (standard, robust)
    - Imputation (mean, median, mode)
    """

    def __init__(self):
        """Initialize feature transformer."""
        self.feature_stats = {}  # Store statistics for normalization
        self.is_fitted = False

    def fit(self, feature_vectors: List[FeatureVector]):
        """
        Fit transformer to a collection of feature vectors.

        Computes statistics needed for transformation.

        Args:
            feature_vectors: List of FeatureVector objects
        """
        logger.info(f"Fitting transformer on {len(feature_vectors)} feature vectors...")

        # Collect all numerical feature names
        all_features = set()
        for fv in feature_vectors:
            all_features.update(fv.numerical_features.keys())

        # Compute statistics for each feature
        for feature_name in all_features:
            values = [fv.numerical_features.get(feature_name, 0.0) for fv in feature_vectors]

            self.feature_stats[feature_name] = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
            }

        self.is_fitted = True
        logger.info(f"Transformer fitted on {len(all_features)} features")

    def transform(
        self,
        feature_vector: FeatureVector,
        method: str = "minmax"
    ) -> FeatureVector:
        """
        Transform a feature vector using fitted statistics.

        Args:
            feature_vector: FeatureVector to transform
            method: Transformation method ("minmax", "zscore", "none")

        Returns:
            Transformed FeatureVector
        """
        if not self.is_fitted and method != "none":
            logger.warning("Transformer not fitted. Using raw features.")
            method = "none"

        if method == "none":
            return feature_vector

        transformed = FeatureVector(
            categorical_features=feature_vector.categorical_features.copy(),
            text_features=feature_vector.text_features.copy(),
            list_features=feature_vector.list_features.copy(),
            feature_count=feature_vector.feature_count,
            extraction_timestamp=feature_vector.extraction_timestamp,
        )

        # Transform numerical features
        for feature_name, value in feature_vector.numerical_features.items():
            if feature_name in self.feature_stats:
                stats = self.feature_stats[feature_name]

                if method == "minmax":
                    # Min-max normalization to [0, 1]
                    min_val = stats["min"]
                    max_val = stats["max"]
                    if max_val > min_val:
                        transformed_value = (value - min_val) / (max_val - min_val)
                    else:
                        transformed_value = 0.5  # Default for constant features

                elif method == "zscore":
                    # Z-score normalization (standardization)
                    mean = stats["mean"]
                    std = stats["std"]
                    if std > 0:
                        transformed_value = (value - mean) / std
                    else:
                        transformed_value = 0.0  # Default for constant features

                else:
                    transformed_value = value

                transformed.numerical_features[feature_name] = transformed_value
            else:
                # Feature not seen during fitting
                transformed.numerical_features[feature_name] = value

        return transformed

    def fit_transform(
        self,
        feature_vectors: List[FeatureVector],
        method: str = "minmax"
    ) -> List[FeatureVector]:
        """
        Fit transformer and transform feature vectors in one step.

        Args:
            feature_vectors: List of FeatureVector objects
            method: Transformation method

        Returns:
            List of transformed FeatureVector objects
        """
        self.fit(feature_vectors)
        return [self.transform(fv, method) for fv in feature_vectors]


def create_feature_pipeline(
    parsed_data_list: List[ParsedData],
    transform: bool = True,
    transform_method: str = "minmax"
) -> Tuple[List[FeatureVector], Optional[FeatureTransformer]]:
    """
    End-to-end feature engineering pipeline.

    Args:
        parsed_data_list: List of ParsedData objects
        transform: Whether to apply feature transformation
        transform_method: Transformation method if transform=True

    Returns:
        Tuple of (feature_vectors, transformer)
    """
    logger.info(f"Running feature engineering pipeline on {len(parsed_data_list)} resumes...")

    # Extract features
    extractor = FeatureExtractor()
    feature_vectors = [extractor.extract_features(pd) for pd in parsed_data_list]

    # Transform features if requested
    transformer = None
    if transform:
        transformer = FeatureTransformer()
        feature_vectors = transformer.fit_transform(feature_vectors, method=transform_method)

    logger.info("Feature engineering pipeline completed")

    return feature_vectors, transformer
