"""
Advanced NLP Module with spaCy Large Models

Features:
- Entity Recognition (dates, companies, locations)
- Advanced text analysis
- Named Entity Recognition for resumes
- Dependency parsing for complex sentences
- Part-of-speech analysis
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from datetime import datetime
from dateutil import parser as date_parser
import warnings

logger = logging.getLogger(__name__)


class AdvancedNLPProcessor:
    """
    Advanced NLP processor using spaCy large models.

    Provides:
    - Named Entity Recognition
    - Date extraction and normalization
    - Company name extraction
    - Location extraction
    - Skill extraction from unstructured text
    """

    def __init__(self, model_name: str = "en_core_web_lg"):
        """
        Initialize NLP processor with spaCy large model.

        Args:
            model_name: spaCy model to use (en_core_web_lg for English)
        """
        self.nlp = None
        self.model_name = model_name

        try:
            import spacy
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"✅ Loaded spaCy model: {model_name}")
            except OSError:
                logger.warning(f"⚠️  spaCy model '{model_name}' not found")
                logger.warning(f"   Install with: python -m spacy download {model_name}")
                logger.warning("   Falling back to rule-based extraction")
                # Try smaller model as fallback
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("✅ Loaded fallback model: en_core_web_sm")
                except OSError:
                    logger.warning("⚠️  No spaCy models available")
        except ImportError:
            logger.warning("⚠️  spaCy not installed. Advanced NLP features disabled.")
            logger.warning("   Install with: pip install spacy")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.

        Returns:
            Dictionary with entity types and their values
        """
        if not self.nlp:
            return self._extract_entities_rules(text)

        entities = {
            "PERSON": [],
            "ORG": [],  # Organizations/Companies
            "GPE": [],  # Geo-political entities (cities, countries)
            "DATE": [],
            "MONEY": [],
            "PERCENT": [],
            "PRODUCT": [],
            "EVENT": [],
        }

        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)

        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def _extract_entities_rules(self, text: str) -> Dict[str, List[str]]:
        """Fallback rule-based entity extraction."""
        entities = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "DATE": [],
            "MONEY": [],
            "PERCENT": [],
            "PRODUCT": [],
            "EVENT": [],
        }

        # Extract dates with regex
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # 01/15/2020
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b',  # January 2020
            r'\b\d{4}\b',  # 2020
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["DATE"].extend(matches)

        # Extract percentages
        percent_pattern = r'\d+\.?\d*\s*%'
        entities["PERCENT"] = re.findall(percent_pattern, text)

        # Extract money
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        entities["MONEY"] = re.findall(money_pattern, text)

        return entities

    def extract_dates(self, text: str) -> List[datetime]:
        """
        Extract and parse dates from text.

        Returns:
            List of datetime objects
        """
        dates = []

        # Use NER if available
        entities = self.extract_entities(text)
        date_strings = entities.get("DATE", [])

        # Parse dates
        for date_str in date_strings:
            try:
                # Use dateutil parser for flexible parsing
                parsed_date = date_parser.parse(date_str, fuzzy=True)
                dates.append(parsed_date)
            except (ValueError, TypeError):
                continue

        return dates

    def extract_companies(self, text: str) -> List[str]:
        """
        Extract company names from text.

        Returns:
            List of company names
        """
        if not self.nlp:
            return self._extract_companies_rules(text)

        doc = self.nlp(text)

        companies = []
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Filter out common false positives
                if len(ent.text) > 2 and not ent.text.isupper():
                    companies.append(ent.text)

        return list(set(companies))

    def _extract_companies_rules(self, text: str) -> List[str]:
        """Rule-based company extraction."""
        companies = []

        # Common company suffixes
        company_suffixes = [
            r'\bInc\.?',
            r'\bLLC',
            r'\bLtd\.?',
            r'\bCorp\.?',
            r'\bCorporation',
            r'\bCompany',
            r'\bCo\.?',
        ]

        for suffix in company_suffixes:
            pattern = r'([A-Z][A-Za-z0-9\s&]+)\s+' + suffix
            matches = re.findall(pattern, text)
            companies.extend(matches)

        return list(set([c.strip() for c in companies]))

    def extract_locations(self, text: str) -> List[str]:
        """
        Extract location names from text.

        Returns:
            List of locations (cities, countries)
        """
        if not self.nlp:
            return self._extract_locations_rules(text)

        doc = self.nlp(text)

        locations = []
        for ent in doc.ents:
            if ent.label_ == "GPE":  # Geo-political entity
                locations.append(ent.text)

        return list(set(locations))

    def _extract_locations_rules(self, text: str) -> List[str]:
        """Rule-based location extraction."""
        # Common locations in tech industry
        common_locations = [
            "San Francisco", "New York", "London", "Tel Aviv", "Silicon Valley",
            "Seattle", "Austin", "Boston", "Berlin", "Amsterdam", "Toronto",
            "Singapore", "Sydney", "Tokyo", "Bangalore", "Remote"
        ]

        locations = []
        for loc in common_locations:
            if loc.lower() in text.lower():
                locations.append(loc)

        return locations

    def extract_skills_nlp(self, text: str, skill_database: Set[str]) -> List[str]:
        """
        Extract skills using NLP context awareness.

        Args:
            text: Resume text
            skill_database: Set of known skills to match

        Returns:
            List of extracted skills
        """
        if not self.nlp:
            # Fallback to simple matching
            return [skill for skill in skill_database if skill.lower() in text.lower()]

        doc = self.nlp(text)

        extracted_skills = []

        # Check for exact matches
        for skill in skill_database:
            if skill.lower() in text.lower():
                extracted_skills.append(skill)

        # Look for skills in context (near certain keywords)
        skill_context_keywords = ["experience", "proficient", "expert", "skilled", "knowledge"]

        for sent in doc.sents:
            sent_text = sent.text.lower()
            # If sentence contains context keywords, extract technical terms
            if any(kw in sent_text for kw in skill_context_keywords):
                for token in sent:
                    # Look for noun chunks that might be skills
                    if token.pos_ in ["NOUN", "PROPN"]:
                        if token.text in skill_database:
                            extracted_skills.append(token.text)

        return list(set(extracted_skills))

    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze text complexity using NLP.

        Returns:
            Dictionary of complexity metrics
        """
        if not self.nlp:
            # Simplified analysis without spaCy
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s for s in sentences if s.strip()]

            return {
                "avg_sentence_length": len(words) / max(len(sentences), 1),
                "unique_word_ratio": len(set(words)) / max(len(words), 1),
                "lexical_density": 0.5,  # Placeholder
                "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            }

        doc = self.nlp(text)

        # Count different POS types
        nouns = sum(1 for token in doc if token.pos_ == "NOUN")
        verbs = sum(1 for token in doc if token.pos_ == "VERB")
        adjectives = sum(1 for token in doc if token.pos_ == "ADJ")
        total_words = len([token for token in doc if not token.is_punct])

        # Lexical density (content words / total words)
        content_words = nouns + verbs + adjectives
        lexical_density = content_words / max(total_words, 1)

        # Sentence statistics
        sentences = list(doc.sents)
        avg_sent_length = total_words / max(len(sentences), 1)

        # Dependency depth (complexity of sentence structure)
        max_depth = 0
        for sent in sentences:
            for token in sent:
                depth = 0
                current = token
                while current.head != current:
                    depth += 1
                    current = current.head
                max_depth = max(max_depth, depth)

        return {
            "avg_sentence_length": avg_sent_length,
            "lexical_density": lexical_density,
            "max_dependency_depth": float(max_depth),
            "noun_ratio": nouns / max(total_words, 1),
            "verb_ratio": verbs / max(total_words, 1),
            "adjective_ratio": adjectives / max(total_words, 1),
        }

    def extract_experience_dates(self, experience_text: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Extract start and end dates from experience description.

        Returns:
            Tuple of (start_date, end_date)
        """
        dates = self.extract_dates(experience_text)

        if not dates:
            return None, None

        # Sort dates
        dates.sort()

        # Assume first date is start, last is end (or current)
        start_date = dates[0] if dates else None
        end_date = dates[-1] if len(dates) > 1 else None

        # Check for "present" or "current"
        if re.search(r'\b(present|current|now)\b', experience_text.lower()):
            end_date = None  # Still current

        return start_date, end_date


class EntityRecognizer:
    """
    Specialized entity recognizer for resume parsing.

    Focuses on:
    - Job titles
    - Company names
    - Dates (employment periods)
    - Locations
    - Educational institutions
    """

    def __init__(self):
        """Initialize entity recognizer."""
        self.nlp_processor = AdvancedNLPProcessor()

    def recognize_job_title(self, text: str) -> Optional[str]:
        """
        Extract job title from text.

        Uses NER and pattern matching.
        """
        # Common job title patterns
        title_patterns = [
            r'(?:as\s+)?(?:a\s+)?([A-Z][A-Za-z\s]+(?:Engineer|Developer|Manager|Director|Analyst|Scientist|Designer|Architect))',
            r'(?:Position|Role|Title):\s*([A-Z][A-Za-z\s]+)',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()

        # Use NER to find potential titles
        if self.nlp_processor.nlp:
            doc = self.nlp_processor.nlp(text)

            # Look for patterns like "worked as [TITLE]"
            for i, token in enumerate(doc):
                if token.lemma_ in ["work", "serve"] and i + 2 < len(doc):
                    if doc[i + 1].text.lower() == "as":
                        # Extract next few tokens as potential title
                        title_tokens = []
                        for j in range(i + 2, min(i + 6, len(doc))):
                            if doc[j].pos_ in ["NOUN", "PROPN", "ADJ"]:
                                title_tokens.append(doc[j].text)
                            else:
                                break
                        if title_tokens:
                            return " ".join(title_tokens)

        return None

    def recognize_dates_in_context(self, text: str, context: str = "experience") -> List[Tuple[datetime, str]]:
        """
        Recognize dates with their context.

        Returns:
            List of (date, context_label) tuples
        """
        dates_with_context = []

        # Extract dates
        raw_dates = self.nlp_processor.extract_dates(text)

        # Determine context for each date
        for date in raw_dates:
            # Look for context words near the date
            date_str = date.strftime("%Y")
            date_pos = text.find(date_str)

            if date_pos != -1:
                # Get surrounding context (50 chars before and after)
                context_start = max(0, date_pos - 50)
                context_end = min(len(text), date_pos + 50)
                surrounding = text[context_start:context_end].lower()

                if any(word in surrounding for word in ["start", "began", "joined"]):
                    label = "start_date"
                elif any(word in surrounding for word in ["end", "left", "until"]):
                    label = "end_date"
                elif any(word in surrounding for word in ["graduate", "completed", "earned"]):
                    label = "graduation_date"
                else:
                    label = "unknown"

                dates_with_context.append((date, label))

        return dates_with_context


# Utility function
def process_resume_with_advanced_nlp(resume_text: str) -> Dict[str, Any]:
    """
    Process entire resume with advanced NLP.

    Returns:
        Dictionary of extracted information
    """
    processor = AdvancedNLPProcessor()
    entity_recognizer = EntityRecognizer()

    result = {
        "entities": processor.extract_entities(resume_text),
        "dates": processor.extract_dates(resume_text),
        "companies": processor.extract_companies(resume_text),
        "locations": processor.extract_locations(resume_text),
        "text_complexity": processor.analyze_text_complexity(resume_text),
    }

    return result


# Import numpy for complexity calculations
try:
    import numpy as np
except ImportError:
    # Fallback implementations
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
