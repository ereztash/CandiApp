"""
CandiApp - AI-Powered Resume Parsing & Screening System

A modern resume parsing and candidate screening system with industry-leading benchmarks.

Version 2.0 Features:
- 219+ engineered features (was 127)
- Advanced NLP with spaCy large models
- Semantic skill matching with embeddings
- BERT-based features and contextual embeddings
- Enhanced parsing with entity recognition
- Multilingual support (English, Spanish, French, German, Hebrew)
- Docker deployment ready
"""

__version__ = "2.0.0"
__author__ = "CandiApp Team"

# Core modules
from .parser import ResumeParser
from .models import Resume, ParsedData
from .scoring import CandidateScorer

# Feature engineering
from .features import (
    FeatureExtractor,
    FeatureTransformer,
    FeatureVector,
    create_feature_pipeline,
)
from .feature_store import FeatureStore, FeatureIndex

# Advanced features (v2.0+)
try:
    from .advanced_features import AdvancedFeatureExtractor
    _has_advanced_features = True
except ImportError:
    _has_advanced_features = False

# Advanced NLP (v2.0+)
try:
    from .nlp_advanced import AdvancedNLPProcessor, EntityRecognizer
    _has_advanced_nlp = True
except ImportError:
    _has_advanced_nlp = False

# Semantic matching (v2.0+)
try:
    from .semantic_matching import SemanticSkillMatcher, calculate_semantic_skill_match_score
    _has_semantic_matching = True
except ImportError:
    _has_semantic_matching = False

# BERT features (v2.0+)
try:
    from .bert_features import BERTFeatureExtractor, TransformerResumeClassifier
    _has_bert = True
except ImportError:
    _has_bert = False

# Enhanced parser (v2.0+)
try:
    from .enhanced_parser import (
        EnhancedExperienceParser,
        EnhancedEducationParser,
        parse_resume_enhanced,
    )
    _has_enhanced_parser = True
except ImportError:
    _has_enhanced_parser = False

# Multilingual (v2.0+)
try:
    from .multilingual import (
        MultilingualProcessor,
        SupportedLanguage,
        detect_resume_language,
    )
    _has_multilingual = True
except ImportError:
    _has_multilingual = False

# Base exports
__all__ = [
    # Core
    "ResumeParser",
    "Resume",
    "ParsedData",
    "CandidateScorer",
    # Feature engineering
    "FeatureExtractor",
    "FeatureTransformer",
    "FeatureVector",
    "create_feature_pipeline",
    "FeatureStore",
    "FeatureIndex",
]

# Advanced exports (v2.0+)
if _has_advanced_features:
    __all__.extend([
        "AdvancedFeatureExtractor",
    ])

if _has_advanced_nlp:
    __all__.extend([
        "AdvancedNLPProcessor",
        "EntityRecognizer",
    ])

if _has_semantic_matching:
    __all__.extend([
        "SemanticSkillMatcher",
        "calculate_semantic_skill_match_score",
    ])

if _has_bert:
    __all__.extend([
        "BERTFeatureExtractor",
        "TransformerResumeClassifier",
    ])

if _has_enhanced_parser:
    __all__.extend([
        "EnhancedExperienceParser",
        "EnhancedEducationParser",
        "parse_resume_enhanced",
    ])

if _has_multilingual:
    __all__.extend([
        "MultilingualProcessor",
        "SupportedLanguage",
        "detect_resume_language",
    ])

# Feature availability flags
features = {
    "advanced_features": _has_advanced_features,
    "advanced_nlp": _has_advanced_nlp,
    "semantic_matching": _has_semantic_matching,
    "bert": _has_bert,
    "enhanced_parser": _has_enhanced_parser,
    "multilingual": _has_multilingual,
}


def get_version_info():
    """Get version and feature availability information."""
    return {
        "version": __version__,
        "features": features,
        "feature_count": 219 if _has_advanced_features else 127,
    }
