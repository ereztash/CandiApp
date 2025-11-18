# Feature Engineering Guide

## Overview

The CandiApp feature engineering module provides comprehensive tools for extracting, transforming, and storing features from parsed resume data. This enables advanced candidate matching, ML-based scoring, and analytics.

## Architecture

```
┌─────────────────┐
│  Resume Parser  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Parsed Data    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│Feature Extractor├─────►│  Feature Vector  │
└─────────────────┘      └────────┬─────────┘
                                  │
                         ┌────────┴─────────┐
                         │                  │
                         ▼                  ▼
                  ┌──────────────┐   ┌─────────────┐
                  │Feature       │   │Feature      │
                  │Transformer   │   │Store        │
                  └──────────────┘   └─────────────┘
```

## Key Components

### 1. Feature Extractor

Extracts 100+ engineered features from parsed resume data.

#### Feature Categories

**Experience Features** (exp_*)
- `exp_count`: Number of work experiences
- `exp_total_years`: Total years of experience
- `exp_avg_tenure`: Average job tenure
- `exp_currently_employed`: Currently employed (1.0 or 0.0)
- `exp_company_diversity`: Number of unique companies
- `exp_job_frequency`: Jobs per year (job hopping indicator)
- `exp_seniority_trend`: Career progression indicator (-1 to 1)
- `exp_total_responsibilities`: Total responsibilities listed
- `exp_total_achievements`: Total achievements listed

**Education Features** (edu_*)
- `edu_count`: Number of education entries
- `edu_max_gpa`: Highest GPA
- `edu_has_stem`: Has STEM degree (1.0 or 0.0)
- `edu_institution_diversity`: Number of unique institutions
- `edu_level_*`: One-hot encoded education levels

**Skills Features** (skills_*)
- `skills_count`: Total skills listed
- `skills_technical_count`: Number of technical skills
- `skills_soft_count`: Number of soft skills
- `skills_category_*`: Skills by category (programming, web, data, ml_ai, cloud, etc.)
- `skills_avg_years`: Average years of experience with skills
- `skills_category_diversity`: Number of skill categories present

**Text Features** (text_*)
- `text_summary_length`: Summary text length
- `text_total_word_count`: Total words in resume
- `text_hebrew_ratio`: Ratio of Hebrew characters
- `text_english_ratio`: Ratio of English characters
- `text_has_*`: Contact information presence (email, phone, linkedin, github)

**Completeness Features** (completeness_*)
- `completeness_score`: Resume completeness score (0-100)
- `completeness_raw_score`: Raw completeness score

**Temporal Features** (temporal_*)
- `temporal_most_recent_exp_year`: Year of most recent experience
- `temporal_years_since_last_exp`: Years since last employment
- `temporal_gap_count`: Number of career gaps
- `temporal_total_gap_years`: Total years in gaps

**Diversity Features** (diversity_*)
- `diversity_language_count`: Number of languages
- `diversity_certification_count`: Number of certifications
- `diversity_project_count`: Number of projects
- `diversity_section_count`: Number of additional sections

#### Usage

```python
from candiapp import FeatureExtractor, ParsedData

# Create extractor
extractor = FeatureExtractor()

# Extract features from parsed data
features = extractor.extract_features(parsed_data)

# Access numerical features
total_exp = features.numerical_features["exp_total_years"]
completeness = features.numerical_features["completeness_score"]

# Access categorical features
exp_level = features.categorical_features.get("exp_level")

# Access list features
all_skills = features.list_features.get("skills_all", [])
```

### 2. Feature Transformer

Transforms and normalizes features for machine learning models.

#### Transformation Methods

**Min-Max Normalization** (`minmax`)
- Scales features to [0, 1] range
- Formula: `(x - min) / (max - min)`
- Best for: Neural networks, distance-based algorithms

**Z-Score Standardization** (`zscore`)
- Centers features around mean with unit variance
- Formula: `(x - mean) / std`
- Best for: Linear models, PCA, clustering

**No Transformation** (`none`)
- Keeps original feature values
- Best for: Tree-based models, exploratory analysis

#### Usage

```python
from candiapp import FeatureTransformer

# Create transformer
transformer = FeatureTransformer()

# Fit on multiple feature vectors
transformer.fit(feature_vectors)

# Transform a single vector
transformed = transformer.transform(feature_vector, method="minmax")

# Or fit and transform in one step
transformed_vectors = transformer.fit_transform(feature_vectors, method="zscore")
```

### 3. Feature Store

Persistent storage for engineered features with caching.

#### Storage Backends

**JSON Backend** (`backend="json"`)
- Human-readable format
- Easy to inspect and debug
- Slightly larger file size
- Compatible across platforms

**Pickle Backend** (`backend="pickle"`)
- Binary format
- Faster serialization
- Smaller file size
- Python-specific

#### Usage

```python
from candiapp import FeatureStore

# Initialize store
store = FeatureStore(
    storage_path="./feature_store",
    backend="json"  # or "pickle"
)

# Save features
store.save("resume_001", feature_vector)

# Load features
features = store.load("resume_001")

# Batch operations
store.batch_save({
    "resume_001": features1,
    "resume_002": features2,
    "resume_003": features3,
})

loaded = store.batch_load(["resume_001", "resume_002"])

# Get statistics
stats = store.get_stats()
print(f"Total features: {stats['total_features']}")
print(f"Storage size: {stats['total_storage_mb']:.2f} MB")

# List all stored features
all_ids = store.list_all()
```

### 4. Feature Index

Fast in-memory index for feature search and analysis.

#### Usage

```python
from candiapp import FeatureIndex

# Create index
index = FeatureIndex()

# Add features to index
index.add("resume_001", feature_vector)

# Search for resumes with specific features
matching = index.search_by_features([
    "exp_total_years",
    "skills_category_programming"
])

# Get all unique feature names
all_features = index.get_all_feature_names()

# Get feature coverage statistics
coverage = index.get_feature_coverage()
for feature, count in sorted(coverage.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feature}: {count} resumes")
```

### 5. Feature Pipeline

End-to-end pipeline for feature engineering.

#### Usage

```python
from candiapp import create_feature_pipeline

# Run complete pipeline
feature_vectors, transformer = create_feature_pipeline(
    parsed_data_list,
    transform=True,
    transform_method="minmax"
)

# Convert to numpy arrays for ML
import numpy as np

feature_names = sorted(feature_vectors[0].numerical_features.keys())
arrays = [fv.to_array(feature_names) for fv in feature_vectors]
X = np.array(arrays)

# Now ready for scikit-learn, TensorFlow, etc.
```

## Complete Example

```python
from candiapp import (
    ResumeParser,
    FeatureExtractor,
    FeatureStore,
    create_feature_pipeline,
)

# 1. Parse resumes
parser = ResumeParser()
resumes = [
    parser.parse("resume1.pdf"),
    parser.parse("resume2.pdf"),
    parser.parse("resume3.pdf"),
]

# 2. Extract parsed data
parsed_data_list = [r.parsed_data for r in resumes if r.is_parsed()]

# 3. Run feature engineering pipeline
feature_vectors, transformer = create_feature_pipeline(
    parsed_data_list,
    transform=True,
    transform_method="minmax"
)

# 4. Store features for later use
store = FeatureStore(storage_path="./features", backend="json")
for i, fv in enumerate(feature_vectors):
    store.save(f"resume_{i+1:03d}", fv)

# 5. Later: Load and use features
loaded_features = store.load("resume_001")
feature_array = loaded_features.to_array()

print(f"Extracted {loaded_features.feature_count} features")
print(f"Completeness: {loaded_features.numerical_features['completeness_score']:.1f}%")
```

## Advanced Topics

### Custom Feature Extraction

You can extend the `FeatureExtractor` class to add custom features:

```python
from candiapp.features import FeatureExtractor

class CustomFeatureExtractor(FeatureExtractor):
    def extract_features(self, parsed_data):
        # Get base features
        features = super().extract_features(parsed_data)

        # Add custom features
        features.numerical_features["custom_metric"] = self._calculate_custom_metric(parsed_data)

        return features

    def _calculate_custom_metric(self, parsed_data):
        # Your custom logic here
        return 42.0
```

### Feature Selection

Select most important features for your use case:

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Convert features to arrays
X = np.array([fv.to_array(feature_names) for fv in feature_vectors])
y = np.array(target_labels)

# Select top 50 features
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = [feature_names[i] for i in selector.get_support(indices=True)]
```

### Performance Tips

1. **Batch Processing**: Use batch operations when processing multiple resumes
2. **Caching**: FeatureStore automatically caches loaded features
3. **Backend Choice**: Use pickle backend for faster I/O, JSON for debugging
4. **Transform Once**: Fit transformer once on training data, reuse for new data

### Integration with Scoring

Features can be used directly in the scoring system:

```python
from candiapp import CandidateScorer, JobRequirements

# Extract features
extractor = FeatureExtractor()
features = extractor.extract_features(parsed_data)

# Use completeness score in ranking
completeness = features.numerical_features["completeness_score"]

# Use skill features for matching
skill_diversity = features.numerical_features["skills_category_diversity"]
```

## Feature Reference

See the full [Feature Reference](./feature_reference.md) for a complete list of all 100+ features with descriptions and examples.

## Best Practices

1. **Always Transform**: Normalize features before using in ML models
2. **Store Features**: Cache extracted features to avoid recomputation
3. **Version Control**: Include feature extraction version in metadata
4. **Monitor Coverage**: Check feature coverage across your dataset
5. **Document Custom Features**: If adding custom features, document them thoroughly

## Troubleshooting

**Issue**: Features have unexpected values

**Solution**: Check if normalization was applied. Use `method="none"` to see raw values.

---

**Issue**: Some features are missing

**Solution**: Features are only extracted when data is available. Check parsed_data completeness.

---

**Issue**: Poor model performance

**Solution**: Try different normalization methods, feature selection, or add domain-specific features.

## Related Documentation

- [Resume Parser Guide](./parser.md)
- [Scoring System Guide](./scoring.md)
- [API Reference](./api_reference.md)
- [Benchmarks](./benchmarks.md)
