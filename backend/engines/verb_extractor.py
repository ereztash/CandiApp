"""
Verb Extractor Engine
Extracts action verbs from resume text and maps them to process archetypes.
Supports both Hebrew and English text.
"""
import re
import json
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
from pathlib import Path
import logging

from models.archetype import (
    ProcessArchetype,
    ArchetypeScore,
    ArchetypeProfile,
    ARCHETYPE_TAXONOMY
)

logger = logging.getLogger(__name__)


class VerbExtractor:
    """
    Extracts action verbs from resume text and maps them to process archetypes.
    Uses linguistic pattern matching and context analysis.
    """

    def __init__(self, taxonomy_path: str = None):
        """
        Initialize the VerbExtractor.

        Args:
            taxonomy_path: Path to verb taxonomy JSON file. If None, uses built-in taxonomy.
        """
        self.taxonomy = self._load_taxonomy(taxonomy_path)
        self.verb_to_archetype_map = self._build_verb_map()
        self.context_patterns = self._build_context_patterns()

        logger.info(f"VerbExtractor initialized with {len(self.taxonomy)} archetypes")

    def _load_taxonomy(self, taxonomy_path: str = None) -> Dict:
        """Load verb taxonomy from JSON file or use built-in."""
        if taxonomy_path and Path(taxonomy_path).exists():
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Use built-in taxonomy from models
            taxonomy = {}
            for archetype, definition in ARCHETYPE_TAXONOMY.items():
                taxonomy[archetype.value] = {
                    "hebrew": definition.hebrew_verbs,
                    "english": definition.english_verbs,
                    "weight": definition.weight,
                    "framenet": definition.framenet_frames,
                    "context_markers": definition.context_markers
                }
            return taxonomy

    def _build_verb_map(self) -> Dict[str, ProcessArchetype]:
        """Build a mapping from verbs to archetypes."""
        verb_map = {}
        for archetype_name, data in self.taxonomy.items():
            archetype = ProcessArchetype(archetype_name)
            # Map Hebrew verbs
            for verb in data["hebrew"]:
                verb_map[verb.lower()] = archetype
            # Map English verbs
            for verb in data["english"]:
                verb_map[verb.lower()] = archetype
        return verb_map

    def _build_context_patterns(self) -> Dict[ProcessArchetype, List[str]]:
        """Build context marker patterns for each archetype."""
        patterns = {}
        for archetype_name, data in self.taxonomy.items():
            archetype = ProcessArchetype(archetype_name)
            patterns[archetype] = [
                marker.lower() for marker in data.get("context_markers", [])
            ]
        return patterns

    def extract_verbs(self, text: str) -> Tuple[List[str], str]:
        """
        Extract all verbs from text.

        Args:
            text: Resume text to analyze

        Returns:
            Tuple of (list of found verbs, dominant language)
        """
        text_lower = text.lower()
        found_verbs = []
        hebrew_count = 0
        english_count = 0

        # Extract verbs by matching against taxonomy
        for verb, archetype in self.verb_to_archetype_map.items():
            # Use word boundary matching for better accuracy
            pattern = r'\b' + re.escape(verb) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                found_verbs.extend([(verb, archetype)] * len(matches))
                # Detect language
                if self._is_hebrew(verb):
                    hebrew_count += len(matches)
                else:
                    english_count += len(matches)

        # Determine dominant language
        if hebrew_count > english_count:
            dominant_language = "hebrew"
        elif english_count > hebrew_count:
            dominant_language = "english"
        else:
            dominant_language = "mixed"

        logger.debug(
            f"Extracted {len(found_verbs)} verbs "
            f"(Hebrew: {hebrew_count}, English: {english_count})"
        )

        return found_verbs, dominant_language

    def _is_hebrew(self, text: str) -> bool:
        """Check if text contains Hebrew characters."""
        return bool(re.search(r'[\u0590-\u05FF]', text))

    def analyze_context(
        self,
        text: str,
        archetype: ProcessArchetype
    ) -> List[str]:
        """
        Find context markers for a specific archetype in text.

        Args:
            text: Resume text
            archetype: Archetype to check context for

        Returns:
            List of found context markers
        """
        text_lower = text.lower()
        markers = self.context_patterns.get(archetype, [])
        found_markers = []

        for marker in markers:
            if marker in text_lower:
                found_markers.append(marker)

        return found_markers

    def calculate_archetype_scores(
        self,
        found_verbs: List[Tuple[str, ProcessArchetype]],
        text: str
    ) -> Dict[ProcessArchetype, ArchetypeScore]:
        """
        Calculate scores for each archetype based on found verbs and context.

        Args:
            found_verbs: List of (verb, archetype) tuples
            text: Original resume text for context analysis

        Returns:
            Dictionary mapping archetypes to their scores
        """
        if not found_verbs:
            # Return zero scores for all archetypes
            return {
                archetype: ArchetypeScore(
                    archetype=archetype,
                    score=0.0,
                    confidence=0.0,
                    evidence_verbs=[],
                    evidence_count=0,
                    context_matches=[]
                )
                for archetype in ProcessArchetype
            }

        # Count verbs per archetype
        archetype_verb_counts = Counter(
            archetype for _, archetype in found_verbs
        )

        # Get unique verbs per archetype
        archetype_unique_verbs = defaultdict(set)
        for verb, archetype in found_verbs:
            archetype_unique_verbs[archetype].add(verb)

        total_verbs = len(found_verbs)
        scores = {}

        for archetype in ProcessArchetype:
            verb_count = archetype_verb_counts.get(archetype, 0)
            unique_verbs = list(archetype_unique_verbs.get(archetype, set()))

            # Find context markers
            context_matches = self.analyze_context(text, archetype)

            # Calculate raw score (frequency-based)
            raw_score = verb_count / total_verbs if total_verbs > 0 else 0.0

            # Apply archetype weight from taxonomy
            archetype_weight = self.taxonomy[archetype.value]["weight"]
            weighted_score = raw_score * archetype_weight

            # Calculate confidence based on:
            # 1. Number of unique verbs (diversity)
            # 2. Total verb count (quantity)
            # 3. Context markers (quality)
            diversity_factor = min(len(unique_verbs) / 5, 1.0)  # Max at 5 unique verbs
            quantity_factor = min(verb_count / 10, 1.0)  # Max at 10 total verbs
            context_factor = min(len(context_matches) / 3, 1.0)  # Max at 3 context matches

            confidence = (
                diversity_factor * 0.4 +
                quantity_factor * 0.3 +
                context_factor * 0.3
            )

            scores[archetype] = ArchetypeScore(
                archetype=archetype,
                score=weighted_score,
                confidence=confidence,
                evidence_verbs=unique_verbs,
                evidence_count=verb_count,
                context_matches=context_matches
            )

        return scores

    def profile_resume(self, resume_text: str) -> ArchetypeProfile:
        """
        Generate a complete archetype profile for a resume.

        Args:
            resume_text: Full resume text

        Returns:
            ArchetypeProfile with primary/secondary archetypes and all scores
        """
        if not resume_text or len(resume_text.strip()) < 10:
            raise ValueError("Resume text is too short or empty")

        # Extract verbs
        found_verbs, dominant_language = self.extract_verbs(resume_text)

        if not found_verbs:
            logger.warning("No process verbs found in resume")
            # Return a default profile with Enabler as default
            return ArchetypeProfile(
                primary_archetype=ProcessArchetype.ENABLER,
                primary_score=0.0,
                primary_confidence=0.0,
                secondary_archetype=None,
                secondary_score=None,
                all_scores={},
                total_verbs_found=0,
                dominant_language=dominant_language
            )

        # Calculate scores for all archetypes
        all_scores = self.calculate_archetype_scores(found_verbs, resume_text)

        # Sort archetypes by score
        sorted_archetypes = sorted(
            all_scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )

        # Determine primary archetype
        primary_archetype, primary_score_obj = sorted_archetypes[0]
        primary_score = primary_score_obj.score
        primary_confidence = primary_score_obj.confidence

        # Determine secondary archetype (if significantly strong)
        secondary_archetype = None
        secondary_score = None
        if len(sorted_archetypes) > 1:
            second_archetype, second_score_obj = sorted_archetypes[1]
            # Only consider as secondary if score is > 20% of primary
            if second_score_obj.score > 0.2 * primary_score:
                secondary_archetype = second_archetype
                secondary_score = second_score_obj.score

        profile = ArchetypeProfile(
            primary_archetype=primary_archetype,
            primary_score=primary_score,
            primary_confidence=primary_confidence,
            secondary_archetype=secondary_archetype,
            secondary_score=secondary_score,
            all_scores=all_scores,
            total_verbs_found=len(found_verbs),
            dominant_language=dominant_language
        )

        logger.info(
            f"Profile generated: Primary={primary_archetype.value} "
            f"({primary_score:.2f}), Verbs={len(found_verbs)}"
        )

        return profile

    def get_archetype_evidence(
        self,
        profile: ArchetypeProfile,
        archetype: ProcessArchetype
    ) -> Tuple[List[str], List[str]]:
        """
        Get evidence (verbs and context) for a specific archetype.

        Args:
            profile: Generated archetype profile
            archetype: Archetype to get evidence for

        Returns:
            Tuple of (evidence verbs, context markers)
        """
        score = profile.all_scores.get(archetype)
        if not score:
            return [], []

        return score.evidence_verbs, score.context_matches
