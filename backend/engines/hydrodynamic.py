"""
Hydrodynamic Controller Engine
Calculates text complexity (Reynolds number) to decide processing level.
Optimizes compute costs by routing simple resumes to fast processing.
"""
import re
import math
from typing import Tuple, Dict
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProcessingLevel(str, Enum):
    """Processing depth levels."""
    LEVEL_1_DEEP = "LEVEL_1_DEEP"  # Deep, thorough analysis
    LEVEL_2_FAST = "LEVEL_2_FAST"  # Fast, pattern-based analysis


@dataclass
class ComplexityMetrics:
    """Text complexity metrics."""
    reynolds_number: float
    vocabulary_richness: float  # Type-Token Ratio
    sentence_complexity: float  # Avg words per sentence
    technical_density: float  # Technical terms ratio
    text_length: int  # Total characters
    recommended_level: ProcessingLevel


class HydrodynamicController:
    """
    Calculates text complexity using fluid dynamics-inspired metrics.
    Routes resumes to appropriate processing level based on complexity.
    """

    def __init__(
        self,
        reynolds_threshold_low: float = 50.0,
        reynolds_threshold_high: float = 200.0
    ):
        """
        Initialize the controller.

        Args:
            reynolds_threshold_low: Below this, use Level 1 (deep)
            reynolds_threshold_high: Above this, use Level 2 (fast)
        """
        self.reynolds_threshold_low = reynolds_threshold_low
        self.reynolds_threshold_high = reynolds_threshold_high

        # Common technical terms (can be expanded)
        self.technical_terms = {
            # Programming
            'python', 'java', 'javascript', 'typescript', 'react', 'angular',
            'vue', 'node', 'django', 'flask', 'fastapi', 'spring', 'docker',
            'kubernetes', 'aws', 'azure', 'gcp', 'microservices', 'api',
            'rest', 'graphql', 'sql', 'nosql', 'mongodb', 'postgresql',
            'redis', 'kafka', 'rabbitmq', 'ci/cd', 'devops', 'agile', 'scrum',
            # Hebrew technical terms
            'תכנות', 'פיתוח', 'תוכנה', 'מערכת', 'ענן', 'מסדנתונים',
            'אינטגרציה', 'ארכיטקטורה', 'אבטחת מידע', 'בינה מלאכותית'
        }

        logger.info(
            f"HydrodynamicController initialized "
            f"(thresholds: {reynolds_threshold_low}, {reynolds_threshold_high})"
        )

    def calculate_complexity(self, text: str) -> ComplexityMetrics:
        """
        Calculate text complexity metrics.

        Args:
            text: Resume text to analyze

        Returns:
            ComplexityMetrics with Reynolds number and recommendation
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text is too short to analyze")

        text_clean = text.strip()
        text_length = len(text_clean)

        # Calculate vocabulary richness (Type-Token Ratio)
        vocabulary_richness = self._calculate_vocabulary_richness(text_clean)

        # Calculate sentence complexity
        sentence_complexity = self._calculate_sentence_complexity(text_clean)

        # Calculate technical density
        technical_density = self._calculate_technical_density(text_clean)

        # Calculate Reynolds number
        reynolds_number = self._calculate_reynolds_number(
            text_length=text_length,
            vocabulary_richness=vocabulary_richness,
            sentence_complexity=sentence_complexity,
            technical_density=technical_density
        )

        # Determine processing level
        recommended_level = self._determine_processing_level(reynolds_number)

        metrics = ComplexityMetrics(
            reynolds_number=reynolds_number,
            vocabulary_richness=vocabulary_richness,
            sentence_complexity=sentence_complexity,
            technical_density=technical_density,
            text_length=text_length,
            recommended_level=recommended_level
        )

        logger.info(
            f"Complexity analysis: Re={reynolds_number:.2f}, "
            f"Level={recommended_level.value}"
        )

        return metrics

    def _calculate_vocabulary_richness(self, text: str) -> float:
        """
        Calculate Type-Token Ratio (vocabulary diversity).

        TTR = (unique words) / (total words)
        Higher values indicate more diverse vocabulary.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        unique_words = set(words)
        ttr = len(unique_words) / len(words)

        # Normalize to 0-1 scale (typical TTR ranges 0.4-0.9)
        normalized_ttr = min(ttr / 0.9, 1.0)

        return normalized_ttr

    def _calculate_sentence_complexity(self, text: str) -> float:
        """
        Calculate average sentence complexity based on words per sentence.

        Higher values indicate more complex sentence structures.
        """
        # Split by sentence terminators
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        total_words = 0
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            total_words += len(words)

        avg_words_per_sentence = total_words / len(sentences)

        # Normalize: typical range is 10-30 words per sentence
        # Complex resumes tend to have 15-25 words per sentence
        normalized = min(avg_words_per_sentence / 30.0, 1.0)

        return normalized

    def _calculate_technical_density(self, text: str) -> float:
        """
        Calculate ratio of technical terms to total words.

        Higher values indicate more technical content.
        """
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        technical_word_count = sum(
            1 for word in words if word in self.technical_terms
        )

        technical_ratio = technical_word_count / len(words)

        # Normalize: typical technical resumes have 5-20% technical terms
        normalized = min(technical_ratio / 0.2, 1.0)

        return normalized

    def _calculate_reynolds_number(
        self,
        text_length: int,
        vocabulary_richness: float,
        sentence_complexity: float,
        technical_density: float
    ) -> float:
        """
        Calculate Reynolds number (complexity metric).

        Inspired by fluid dynamics Reynolds number:
        Re = (ρ * v * L) / μ

        Where:
        - ρ (density) ≈ vocabulary_richness (information density)
        - v (velocity) ≈ sentence_complexity (information flow)
        - L (characteristic length) ≈ log(text_length) (scale)
        - μ (viscosity) ≈ 1 / technical_density (resistance to understanding)

        Returns:
            Reynolds number (higher = more complex, needs deep processing)
        """
        # Avoid division by zero
        viscosity = 1.0 / max(technical_density, 0.01)

        # Use log of text length to prevent extremely high values
        characteristic_length = math.log(max(text_length, 100))

        # Calculate Reynolds number
        reynolds = (
            vocabulary_richness *
            sentence_complexity *
            characteristic_length
        ) / viscosity

        # Scale to reasonable range (typically 0-500)
        reynolds_scaled = reynolds * 10

        return reynolds_scaled

    def _determine_processing_level(
        self,
        reynolds_number: float
    ) -> ProcessingLevel:
        """
        Determine which processing level to use based on Reynolds number.

        Args:
            reynolds_number: Calculated complexity metric

        Returns:
            Recommended processing level
        """
        if reynolds_number < self.reynolds_threshold_low:
            # Low complexity - use fast processing
            return ProcessingLevel.LEVEL_2_FAST
        elif reynolds_number > self.reynolds_threshold_high:
            # High complexity - use deep processing
            return ProcessingLevel.LEVEL_1_DEEP
        else:
            # Medium complexity - use deep processing to be safe
            return ProcessingLevel.LEVEL_1_DEEP

    def should_use_deep_processing(self, text: str) -> Tuple[bool, ComplexityMetrics]:
        """
        Determine if deep processing should be used for given text.

        Args:
            text: Resume text to analyze

        Returns:
            Tuple of (use_deep_processing, complexity_metrics)
        """
        metrics = self.calculate_complexity(text)
        use_deep = metrics.recommended_level == ProcessingLevel.LEVEL_1_DEEP

        return use_deep, metrics

    def get_cost_savings_estimate(self, metrics: ComplexityMetrics) -> Dict[str, float]:
        """
        Estimate cost savings from using recommended processing level.

        Args:
            metrics: Complexity metrics

        Returns:
            Dictionary with cost analysis
        """
        # Assume Level 1 costs 1.0 unit, Level 2 costs 0.3 units
        level_1_cost = 1.0
        level_2_cost = 0.3

        if metrics.recommended_level == ProcessingLevel.LEVEL_1_DEEP:
            actual_cost = level_1_cost
            potential_savings = 0.0
            savings_percent = 0.0
        else:
            actual_cost = level_2_cost
            potential_savings = level_1_cost - level_2_cost
            savings_percent = (potential_savings / level_1_cost) * 100

        return {
            "actual_cost": actual_cost,
            "baseline_cost": level_1_cost,
            "savings": potential_savings,
            "savings_percent": savings_percent,
            "recommended_level": metrics.recommended_level.value
        }
