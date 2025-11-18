"""
Multilingual Support Module

Adds support for multiple languages:
- Spanish (Español)
- French (Français)
- German (Deutsch)
- Hebrew (עברית) - already supported
- English - base language

Features:
- Language detection
- Multi-language NLP models
- Translated keyword matching
- Locale-aware date parsing
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages."""
    ENGLISH = "en"
    HEBREW = "he"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"


class MultilingualProcessor:
    """
    Process resumes in multiple languages.

    Provides:
    - Language detection
    - Multi-language keyword dictionaries
    - Language-specific NLP models
    - Translation support
    """

    # Language-specific keywords for sections
    SECTION_KEYWORDS = {
        SupportedLanguage.ENGLISH: {
            "experience": ["experience", "work history", "employment", "professional experience"],
            "education": ["education", "academic background", "qualifications"],
            "skills": ["skills", "competencies", "technical skills", "expertise"],
            "summary": ["summary", "objective", "profile", "about me"],
            "languages": ["languages", "language skills"],
            "certifications": ["certifications", "certificates", "licenses"],
        },
        SupportedLanguage.SPANISH: {
            "experience": ["experiencia", "experiencia laboral", "historial laboral", "trayectoria profesional"],
            "education": ["educación", "formación académica", "estudios"],
            "skills": ["habilidades", "competencias", "habilidades técnicas", "capacidades"],
            "summary": ["resumen", "objetivo", "perfil profesional", "sobre mí"],
            "languages": ["idiomas", "lenguas"],
            "certifications": ["certificaciones", "certificados", "licencias"],
        },
        SupportedLanguage.FRENCH: {
            "experience": ["expérience", "expérience professionnelle", "parcours professionnel"],
            "education": ["formation", "éducation", "études", "formation académique"],
            "skills": ["compétences", "compétences techniques", "savoir-faire"],
            "summary": ["résumé", "objectif", "profil", "à propos"],
            "languages": ["langues", "langues parlées"],
            "certifications": ["certifications", "certificats", "diplômes"],
        },
        SupportedLanguage.GERMAN: {
            "experience": ["berufserfahrung", "erfahrung", "werdegang", "beruflicher werdegang"],
            "education": ["ausbildung", "bildung", "akademischer werdegang", "studium"],
            "skills": ["fähigkeiten", "kompetenzen", "kenntnisse", "technische fähigkeiten"],
            "summary": ["zusammenfassung", "profil", "über mich", "zielsetzung"],
            "languages": ["sprachen", "sprachkenntnisse"],
            "certifications": ["zertifizierungen", "zertifikate", "lizenzen"],
        },
        SupportedLanguage.HEBREW: {
            "experience": ["ניסיון", "ניסיון תעסוקתי", "עבודה", "קריירה"],
            "education": ["השכלה", "לימודים", "השכלה אקדמית"],
            "skills": ["כישורים", "מיומנויות", "יכולות", "כישורים טכניים"],
            "summary": ["תקציר", "פרופיל", "אודות"],
            "languages": ["שפות", "שפות מדוברות"],
            "certifications": ["תעודות", "הסמכות", "רישיונות"],
        }
    }

    # Job title translations
    JOB_TITLES = {
        SupportedLanguage.ENGLISH: {
            "software engineer": "software engineer",
            "data scientist": "data scientist",
            "product manager": "product manager",
        },
        SupportedLanguage.SPANISH: {
            "ingeniero de software": "software engineer",
            "científico de datos": "data scientist",
            "gerente de producto": "product manager",
        },
        SupportedLanguage.FRENCH: {
            "ingénieur logiciel": "software engineer",
            "scientifique des données": "data scientist",
            "chef de produit": "product manager",
        },
        SupportedLanguage.GERMAN: {
            "softwareentwickler": "software engineer",
            "datenwissenschaftler": "data scientist",
            "produktmanager": "product manager",
        },
    }

    # Skills translations
    SKILLS_TRANSLATIONS = {
        SupportedLanguage.SPANISH: {
            "programación": "programming",
            "desarrollo web": "web development",
            "aprendizaje automático": "machine learning",
            "inteligencia artificial": "artificial intelligence",
        },
        SupportedLanguage.FRENCH: {
            "programmation": "programming",
            "développement web": "web development",
            "apprentissage automatique": "machine learning",
            "intelligence artificielle": "artificial intelligence",
        },
        SupportedLanguage.GERMAN: {
            "programmierung": "programming",
            "webentwicklung": "web development",
            "maschinelles lernen": "machine learning",
            "künstliche intelligenz": "artificial intelligence",
        },
    }

    def __init__(self):
        """Initialize multilingual processor."""
        self.nlp_models = {}
        self._load_language_models()

    def _load_language_models(self):
        """Load spaCy models for supported languages."""
        model_names = {
            SupportedLanguage.ENGLISH: "en_core_web_lg",
            SupportedLanguage.SPANISH: "es_core_news_lg",
            SupportedLanguage.FRENCH: "fr_core_news_lg",
            SupportedLanguage.GERMAN: "de_core_news_lg",
            # Hebrew - would need custom model or use multilingual
        }

        try:
            import spacy

            for lang, model_name in model_names.items():
                try:
                    self.nlp_models[lang] = spacy.load(model_name)
                    logger.info(f"✅ Loaded {lang.value} model: {model_name}")
                except OSError:
                    logger.warning(f"⚠️  Model '{model_name}' not found for {lang.value}")
                    logger.warning(f"   Install with: python -m spacy download {model_name}")

        except ImportError:
            logger.warning("⚠️  spaCy not installed. Multilingual features limited.")

    def detect_language(self, text: str) -> SupportedLanguage:
        """
        Detect the primary language of text.

        Args:
            text: Resume text

        Returns:
            Detected language
        """
        text_lower = text.lower()

        # Count language-specific characters
        language_scores = {
            SupportedLanguage.HEBREW: 0,
            SupportedLanguage.ENGLISH: 0,
            SupportedLanguage.SPANISH: 0,
            SupportedLanguage.FRENCH: 0,
            SupportedLanguage.GERMAN: 0,
        }

        # Hebrew detection (Hebrew characters)
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        language_scores[SupportedLanguage.HEBREW] = hebrew_chars

        # Check for language-specific keywords
        for lang, keywords_dict in self.SECTION_KEYWORDS.items():
            all_keywords = []
            for section_keywords in keywords_dict.values():
                all_keywords.extend(section_keywords)

            # Count keyword matches
            matches = sum(1 for kw in all_keywords if kw in text_lower)
            language_scores[lang] += matches * 100  # Weight keyword matches highly

        # Spanish-specific characters (ñ, á, é, í, ó, ú)
        spanish_chars = len(re.findall(r'[ñáéíóúü]', text_lower))
        language_scores[SupportedLanguage.SPANISH] += spanish_chars * 10

        # French-specific characters (é, è, ê, à, ç)
        french_chars = len(re.findall(r'[éèêàùçœ]', text_lower))
        language_scores[SupportedLanguage.FRENCH] += french_chars * 10

        # German-specific characters (ä, ö, ü, ß)
        german_chars = len(re.findall(r'[äöüß]', text_lower))
        language_scores[SupportedLanguage.GERMAN] += german_chars * 10

        # English as default if no strong signals
        english_words = ["the", "and", "is", "at", "to", "for"]
        english_matches = sum(1 for word in english_words if f" {word} " in text_lower)
        language_scores[SupportedLanguage.ENGLISH] += english_matches * 50

        # Get highest score
        detected_lang = max(language_scores.items(), key=lambda x: x[1])[0]

        logger.info(f"Detected language: {detected_lang.value}")
        return detected_lang

    def get_section_keywords(self, language: SupportedLanguage) -> Dict[str, List[str]]:
        """
        Get section keywords for a language.

        Args:
            language: Target language

        Returns:
            Dictionary of section keywords
        """
        return self.SECTION_KEYWORDS.get(language, self.SECTION_KEYWORDS[SupportedLanguage.ENGLISH])

    def translate_to_english(self, text: str, source_lang: SupportedLanguage) -> str:
        """
        Translate text to English (simplified - would use translation API in production).

        Args:
            text: Text to translate
            source_lang: Source language

        Returns:
            Translated text (or original if translation not available)
        """
        if source_lang == SupportedLanguage.ENGLISH:
            return text

        # This is a placeholder - in production would use:
        # - Google Translate API
        # - DeepL API
        # - MarianMT (free, open-source)
        # - mBART for multilingual translation

        logger.info(f"Translation from {source_lang.value} to English (placeholder)")

        # For now, return original
        # TODO: Implement actual translation
        return text

    def normalize_skill_name(self, skill: str, language: SupportedLanguage) -> str:
        """
        Normalize skill name to English.

        Args:
            skill: Skill name in any language
            language: Source language

        Returns:
            Normalized English skill name
        """
        skill_lower = skill.lower()

        if language == SupportedLanguage.ENGLISH:
            return skill

        # Check translation dictionary
        translations = self.SKILLS_TRANSLATIONS.get(language, {})

        for foreign_skill, english_skill in translations.items():
            if foreign_skill in skill_lower:
                return english_skill

        # Return original if no translation found
        return skill

    def extract_entities_multilingual(
        self,
        text: str,
        language: SupportedLanguage
    ) -> Dict[str, List[str]]:
        """
        Extract entities using language-specific NLP model.

        Args:
            text: Text to process
            language: Text language

        Returns:
            Dictionary of entities
        """
        if language not in self.nlp_models:
            logger.warning(f"No NLP model for {language.value}, using rule-based extraction")
            return self._extract_entities_rules(text, language)

        nlp = self.nlp_models[language]
        doc = nlp(text)

        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
        }

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def _extract_entities_rules(
        self,
        text: str,
        language: SupportedLanguage
    ) -> Dict[str, List[str]]:
        """Fallback rule-based entity extraction for languages without NLP models."""
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
        }

        # Extract dates (universal patterns)
        date_pattern = r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}'
        entities["DATE"] = re.findall(date_pattern, text)

        return entities


class LanguageDetector:
    """
    Fast language detector using character n-grams and keywords.
    """

    # Language-specific character patterns
    LANGUAGE_PATTERNS = {
        SupportedLanguage.HEBREW: r'[\u0590-\u05FF]',
        SupportedLanguage.SPANISH: r'[ñáéíóúü¿¡]',
        SupportedLanguage.FRENCH: r'[àâäéèêëïîôùûüÿœç]',
        SupportedLanguage.GERMAN: r'[äöüßÄÖÜ]',
    }

    # Common words for quick detection
    COMMON_WORDS = {
        SupportedLanguage.ENGLISH: {"the", "and", "is", "to", "for", "of", "in", "with"},
        SupportedLanguage.SPANISH: {"el", "la", "de", "en", "y", "que", "los", "para"},
        SupportedLanguage.FRENCH: {"le", "la", "de", "et", "en", "dans", "pour", "les"},
        SupportedLanguage.GERMAN: {"der", "die", "das", "und", "in", "zu", "den", "ist"},
        SupportedLanguage.HEBREW: {"את", "של", "על", "עם", "אל", "אלה", "אשר"},
    }

    @classmethod
    def detect(cls, text: str) -> SupportedLanguage:
        """
        Fast language detection.

        Args:
            text: Text to analyze

        Returns:
            Detected language
        """
        text_lower = text.lower()

        scores = {lang: 0 for lang in SupportedLanguage}

        # Character pattern matching
        for lang, pattern in cls.LANGUAGE_PATTERNS.items():
            matches = len(re.findall(pattern, text))
            scores[lang] += matches * 10

        # Common word matching
        words = set(text_lower.split())
        for lang, common_words in cls.COMMON_WORDS.items():
            overlap = len(words & common_words)
            scores[lang] += overlap * 50

        # Return language with highest score
        detected = max(scores.items(), key=lambda x: x[1])[0]

        # Default to English if all scores are low
        if scores[detected] < 10:
            detected = SupportedLanguage.ENGLISH

        return detected


# Utility functions
def detect_resume_language(resume_text: str) -> SupportedLanguage:
    """
    Detect resume language.

    Args:
        resume_text: Resume text

    Returns:
        Detected language
    """
    return LanguageDetector.detect(resume_text)


def process_multilingual_resume(resume_text: str) -> Dict[str, any]:
    """
    Process resume in any supported language.

    Args:
        resume_text: Resume text

    Returns:
        Dictionary with language detection and processing results
    """
    processor = MultilingualProcessor()

    # Detect language
    language = processor.detect_language(resume_text)

    # Get language-specific keywords
    keywords = processor.get_section_keywords(language)

    # Extract entities
    entities = processor.extract_entities_multilingual(resume_text, language)

    return {
        "detected_language": language.value,
        "section_keywords": keywords,
        "entities": entities,
    }
