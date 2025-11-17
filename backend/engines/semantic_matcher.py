"""
Semantic Matcher Engine
Matches candidate skills against job requirements.
Uses fuzzy matching and synonym expansion for better accuracy.
"""
import re
from typing import List, Set, Dict, Tuple
from difflib import SequenceMatcher
import logging

from models.candidate import SemanticFitScore

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """
    Matches skills and keywords between resume and job requirements.
    Supports fuzzy matching and common variations/synonyms.
    """

    def __init__(self, fuzzy_threshold: float = 0.85):
        """
        Initialize the SemanticMatcher.

        Args:
            fuzzy_threshold: Minimum similarity ratio for fuzzy matching (0.0-1.0)
        """
        self.fuzzy_threshold = fuzzy_threshold

        # Common skill synonyms and variations
        self.skill_synonyms = {
            # Programming languages
            'javascript': {'js', 'ecmascript', 'node.js', 'nodejs'},
            'typescript': {'ts'},
            'python': {'py'},
            'c++': {'cpp', 'cplusplus'},
            'c#': {'csharp', 'c sharp'},

            # Frameworks
            'react': {'reactjs', 'react.js'},
            'vue': {'vuejs', 'vue.js'},
            'angular': {'angularjs', 'angular.js'},
            'fastapi': {'fast api'},
            'django': {'django rest framework', 'drf'},

            # Databases
            'postgresql': {'postgres', 'psql'},
            'mongodb': {'mongo'},
            'mysql': {'my sql'},

            # Cloud
            'aws': {'amazon web services'},
            'gcp': {'google cloud platform', 'google cloud'},
            'azure': {'microsoft azure'},

            # DevOps
            'kubernetes': {'k8s'},
            'ci/cd': {'continuous integration', 'continuous deployment', 'cicd'},
            'docker': {'containerization', 'containers'},

            # Methodologies
            'agile': {'scrum', 'kanban'},
            'tdd': {'test driven development', 'test-driven development'},

            # Hebrew equivalents
            'פיתוח': {'development', 'dev'},
            'תכנות': {'programming', 'coding'},
            'ענן': {'cloud'},
        }

        # Build reverse synonym map
        self.expanded_synonyms = self._build_expanded_synonyms()

        logger.info(
            f"SemanticMatcher initialized with {len(self.skill_synonyms)} "
            f"synonym groups, threshold={fuzzy_threshold}"
        )

    def _build_expanded_synonyms(self) -> Dict[str, Set[str]]:
        """Build bidirectional synonym mapping."""
        expanded = {}

        for primary, synonyms in self.skill_synonyms.items():
            # Add primary term
            all_terms = {primary} | synonyms

            # Map each term to all other terms
            for term in all_terms:
                expanded[term.lower()] = {t.lower() for t in all_terms}

        return expanded

    def normalize_skill(self, skill: str) -> str:
        """
        Normalize a skill string.

        Args:
            skill: Raw skill string

        Returns:
            Normalized skill string
        """
        # Convert to lowercase
        normalized = skill.lower().strip()

        # Remove special characters (keep alphanumeric, spaces, and hyphens)
        normalized = re.sub(r'[^\w\s\-/#+.]', '', normalized)

        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)

        return normalized

    def extract_skills(self, text: str) -> Set[str]:
        """
        Extract potential skills from text.

        Args:
            text: Resume or job description text

        Returns:
            Set of normalized skills found
        """
        text_lower = text.lower()
        skills_found = set()

        # Look for common patterns
        # Pattern 1: Bullet points with skills
        bullets = re.findall(r'[•\-\*]\s*([^\n•\-\*]+)', text)
        for bullet in bullets:
            # Extract technical terms (capitalized words, acronyms)
            terms = re.findall(r'\b[A-Z][A-Za-z0-9+#.]+\b', bullet)
            skills_found.update(self.normalize_skill(term) for term in terms)

        # Pattern 2: "Skills:" section
        skills_section = re.search(
            r'(?:skills?|technologies|tools?)[\s:]+([^\n]+)',
            text_lower
        )
        if skills_section:
            skills_text = skills_section.group(1)
            # Split by common delimiters
            terms = re.split(r'[,;|/]', skills_text)
            skills_found.update(self.normalize_skill(term) for term in terms if term.strip())

        return skills_found

    def fuzzy_match(self, skill1: str, skill2: str) -> float:
        """
        Calculate fuzzy similarity between two skills.

        Args:
            skill1: First skill string
            skill2: Second skill string

        Returns:
            Similarity ratio (0.0-1.0)
        """
        s1 = self.normalize_skill(skill1)
        s2 = self.normalize_skill(skill2)

        # Exact match
        if s1 == s2:
            return 1.0

        # Check synonyms
        if s1 in self.expanded_synonyms and s2 in self.expanded_synonyms[s1]:
            return 1.0

        # Fuzzy string matching
        similarity = SequenceMatcher(None, s1, s2).ratio()

        return similarity

    def match_skill(
        self,
        target_skill: str,
        candidate_skills: List[str]
    ) -> Tuple[bool, str, float]:
        """
        Try to match a target skill against candidate's skills.

        Args:
            target_skill: Required skill to match
            candidate_skills: List of candidate's skills

        Returns:
            Tuple of (matched, matched_skill, similarity_score)
        """
        target_normalized = self.normalize_skill(target_skill)

        best_match = None
        best_score = 0.0

        for candidate_skill in candidate_skills:
            score = self.fuzzy_match(target_normalized, candidate_skill)

            if score > best_score:
                best_score = score
                best_match = candidate_skill

        # Consider it a match if above threshold
        matched = best_score >= self.fuzzy_threshold

        return matched, best_match or "", best_score

    def match_skills(
        self,
        resume_text: str,
        required_skills: List[str],
        preferred_skills: List[str],
        required_weight: float = 0.8,
        preferred_weight: float = 0.2
    ) -> SemanticFitScore:
        """
        Match resume against required and preferred skills.

        Args:
            resume_text: Full resume text
            required_skills: Must-have skills for the job
            preferred_skills: Nice-to-have skills
            required_weight: Weight for required skills in overall score
            preferred_weight: Weight for preferred skills in overall score

        Returns:
            SemanticFitScore with detailed matching results
        """
        # Extract skills from resume
        resume_text_lower = resume_text.lower()

        # Simple skill matching - check if skill appears in resume
        matched_required = []
        missing_required = []

        for skill in required_skills:
            skill_normalized = self.normalize_skill(skill)

            # Check for direct match or synonyms
            found = False

            # Check exact match
            if skill_normalized in resume_text_lower:
                found = True
            else:
                # Check synonyms
                if skill_normalized in self.expanded_synonyms:
                    for synonym in self.expanded_synonyms[skill_normalized]:
                        if synonym in resume_text_lower:
                            found = True
                            break

            if found:
                matched_required.append(skill)
            else:
                missing_required.append(skill)

        # Match preferred skills
        matched_preferred = []
        missing_preferred = []

        for skill in preferred_skills:
            skill_normalized = self.normalize_skill(skill)

            found = False

            if skill_normalized in resume_text_lower:
                found = True
            else:
                if skill_normalized in self.expanded_synonyms:
                    for synonym in self.expanded_synonyms[skill_normalized]:
                        if synonym in resume_text_lower:
                            found = True
                            break

            if found:
                matched_preferred.append(skill)
            else:
                missing_preferred.append(skill)

        # Calculate match rates
        required_match_rate = (
            len(matched_required) / len(required_skills)
            if required_skills else 1.0
        )
        preferred_match_rate = (
            len(matched_preferred) / len(preferred_skills)
            if preferred_skills else 1.0
        )

        # Calculate overall semantic score
        overall_score = (
            required_match_rate * required_weight +
            preferred_match_rate * preferred_weight
        )

        # Find additional skills (skills in resume but not in requirements)
        # This is a simplified version - in production, use NLP extraction
        additional_skills = []

        result = SemanticFitScore(
            overall_score=overall_score,
            matched_required_skills=matched_required,
            matched_preferred_skills=matched_preferred,
            missing_required_skills=missing_required,
            missing_preferred_skills=missing_preferred,
            required_match_rate=required_match_rate,
            preferred_match_rate=preferred_match_rate,
            additional_skills=additional_skills
        )

        logger.info(
            f"Semantic match: {overall_score:.2%} "
            f"(Required: {required_match_rate:.2%}, "
            f"Preferred: {preferred_match_rate:.2%})"
        )

        return result

    def explain_match(
        self,
        semantic_score: SemanticFitScore,
        required_skills: List[str]
    ) -> str:
        """
        Generate human-readable explanation of skill matching.

        Args:
            semantic_score: Semantic fit score result
            required_skills: Original required skills

        Returns:
            Explanation string
        """
        matched_count = len(semantic_score.matched_required_skills)
        total_required = len(required_skills)
        missing_count = len(semantic_score.missing_required_skills)

        if semantic_score.overall_score >= 0.8:
            explanation = (
                f"Strong skills match ({matched_count}/{total_required} required skills). "
            )
        elif semantic_score.overall_score >= 0.6:
            explanation = (
                f"Good skills match ({matched_count}/{total_required} required skills). "
            )
        else:
            explanation = (
                f"Weak skills match ({matched_count}/{total_required} required skills). "
            )

        if missing_count > 0:
            explanation += (
                f"Missing: {', '.join(semantic_score.missing_required_skills[:3])}"
            )
            if missing_count > 3:
                explanation += f" and {missing_count - 3} more"

        return explanation
