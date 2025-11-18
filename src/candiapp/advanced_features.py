"""
Advanced Feature Engineering Module

Adds 50+ advanced features to reach 180+ total features:
- Industry-specific features
- Behavioral indicators
- Network features (LinkedIn, GitHub activity)
- Achievement analysis
- Keyword extraction
- Writing style analysis
- And more...
"""

import re
import math
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from datetime import datetime, timedelta
import logging

from .models import ParsedData, Experience, Education, Skill

logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """
    Extracts 50+ advanced features beyond the base feature set.

    New feature categories:
    - Industry & domain features
    - Achievement quality metrics
    - Behavioral indicators
    - Network & social features
    - Writing quality features
    - Career trajectory features
    - Specialization metrics
    """

    # Industry keywords
    INDUSTRY_KEYWORDS = {
        "fintech": ["fintech", "banking", "finance", "trading", "payment", "blockchain", "crypto"],
        "healthcare": ["healthcare", "medical", "health", "hospital", "clinical", "pharma"],
        "ecommerce": ["ecommerce", "retail", "marketplace", "shopping", "commerce"],
        "enterprise": ["enterprise", "b2b", "saas", "crm", "erp"],
        "consumer": ["consumer", "b2c", "mobile app", "social"],
        "gaming": ["gaming", "game", "unity", "unreal", "esports"],
        "iot": ["iot", "embedded", "hardware", "firmware", "sensors"],
        "security": ["security", "cybersecurity", "infosec", "penetration", "vulnerability"],
    }

    # Role types
    ROLE_TYPES = {
        "individual_contributor": ["engineer", "developer", "analyst", "scientist", "designer"],
        "manager": ["manager", "director", "head", "vp", "chief"],
        "architect": ["architect", "principal", "staff"],
        "researcher": ["researcher", "scientist", "phd"],
        "consultant": ["consultant", "advisor", "specialist"],
    }

    # Achievement quality indicators
    ACHIEVEMENT_QUALITY_INDICATORS = {
        "quantified": [r"\d+%", r"\d+x", r"\d+\s*times", r"by\s+\d+"],
        "impact": ["improved", "increased", "reduced", "decreased", "optimized", "enhanced"],
        "scale": ["million", "thousand", "users", "customers", "team"],
        "leadership": ["led", "managed", "directed", "coordinated", "mentored"],
        "innovation": ["built", "created", "designed", "launched", "pioneered"],
    }

    # Technical depth indicators
    TECH_DEPTH_KEYWORDS = {
        "architecture": ["architecture", "design pattern", "microservices", "distributed systems"],
        "algorithms": ["algorithm", "data structure", "optimization", "complexity"],
        "systems": ["system design", "scalability", "performance", "infrastructure"],
        "databases": ["database design", "sql optimization", "indexing", "sharding"],
        "security": ["security", "encryption", "authentication", "authorization"],
    }

    def extract_advanced_features(
        self,
        parsed_data: ParsedData,
        base_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Extract 50+ advanced features.

        Args:
            parsed_data: Parsed resume data
            base_features: Existing base features

        Returns:
            Dictionary of advanced features
        """
        features = {}

        # 1. Industry & Domain Features (10 features)
        features.update(self._extract_industry_features(parsed_data))

        # 2. Achievement Quality Features (8 features)
        features.update(self._extract_achievement_quality_features(parsed_data))

        # 3. Behavioral Indicators (7 features)
        features.update(self._extract_behavioral_features(parsed_data))

        # 4. Network & Social Features (5 features)
        features.update(self._extract_network_features(parsed_data))

        # 5. Writing Quality Features (6 features)
        features.update(self._extract_writing_quality_features(parsed_data))

        # 6. Career Trajectory Features (8 features)
        features.update(self._extract_career_trajectory_features(parsed_data))

        # 7. Specialization Features (6 features)
        features.update(self._extract_specialization_features(parsed_data))

        # 8. Education Quality Features (5 features)
        features.update(self._extract_education_quality_features(parsed_data))

        # 9. Experience Depth Features (5 features)
        features.update(self._extract_experience_depth_features(parsed_data))

        logger.info(f"Extracted {len(features)} advanced features")

        return features

    def _extract_industry_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract industry-specific features."""
        features = {}

        # Combine all text
        all_text = " ".join([
            parsed_data.summary or "",
            " ".join([exp.description or "" for exp in parsed_data.experiences]),
            " ".join([exp.title or "" for exp in parsed_data.experiences]),
        ]).lower()

        # Industry detection
        for industry, keywords in self.INDUSTRY_KEYWORDS.items():
            count = sum(all_text.count(kw) for kw in keywords)
            features[f"industry_{industry}"] = float(count > 0)
            features[f"industry_{industry}_strength"] = min(float(count), 5.0)

        # Industry diversity (how many industries)
        industries_present = sum(1 for industry in self.INDUSTRY_KEYWORDS.keys()
                                if features.get(f"industry_{industry}", 0) > 0)
        features["industry_diversity"] = float(industries_present)

        # Industry specialization (focused vs generalist)
        if industries_present > 0:
            max_strength = max(features.get(f"industry_{ind}_strength", 0)
                             for ind in self.INDUSTRY_KEYWORDS.keys())
            features["industry_specialization"] = max_strength / max(industries_present, 1)
        else:
            features["industry_specialization"] = 0.0

        return features

    def _extract_achievement_quality_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract achievement quality metrics."""
        features = {}

        # Collect all achievements
        all_achievements = []
        for exp in parsed_data.experiences:
            all_achievements.extend(exp.achievements)

        if not all_achievements:
            # Default values if no achievements
            features["achievement_quantified_ratio"] = 0.0
            features["achievement_avg_impact_words"] = 0.0
            features["achievement_avg_scale_mentions"] = 0.0
            features["achievement_leadership_ratio"] = 0.0
            features["achievement_innovation_ratio"] = 0.0
            features["achievement_avg_length"] = 0.0
            features["achievement_quality_score"] = 0.0
            features["achievement_specificity"] = 0.0
            return features

        # Analyze achievements
        quantified_count = 0
        impact_scores = []
        scale_scores = []
        leadership_count = 0
        innovation_count = 0
        lengths = []

        for achievement in all_achievements:
            text = achievement.lower()
            lengths.append(len(achievement.split()))

            # Quantified achievements (contains numbers)
            if any(re.search(pattern, text) for pattern in self.ACHIEVEMENT_QUALITY_INDICATORS["quantified"]):
                quantified_count += 1

            # Impact words
            impact_count = sum(text.count(word) for word in self.ACHIEVEMENT_QUALITY_INDICATORS["impact"])
            impact_scores.append(impact_count)

            # Scale mentions
            scale_count = sum(text.count(word) for word in self.ACHIEVEMENT_QUALITY_INDICATORS["scale"])
            scale_scores.append(scale_count)

            # Leadership
            if any(word in text for word in self.ACHIEVEMENT_QUALITY_INDICATORS["leadership"]):
                leadership_count += 1

            # Innovation
            if any(word in text for word in self.ACHIEVEMENT_QUALITY_INDICATORS["innovation"]):
                innovation_count += 1

        total = len(all_achievements)

        features["achievement_quantified_ratio"] = quantified_count / total
        features["achievement_avg_impact_words"] = np.mean(impact_scores) if impact_scores else 0.0
        features["achievement_avg_scale_mentions"] = np.mean(scale_scores) if scale_scores else 0.0
        features["achievement_leadership_ratio"] = leadership_count / total
        features["achievement_innovation_ratio"] = innovation_count / total
        features["achievement_avg_length"] = np.mean(lengths) if lengths else 0.0

        # Overall quality score (0-100)
        quality_score = (
            features["achievement_quantified_ratio"] * 30 +
            min(features["achievement_avg_impact_words"], 3) / 3 * 25 +
            features["achievement_leadership_ratio"] * 20 +
            features["achievement_innovation_ratio"] * 15 +
            min(features["achievement_avg_length"] / 15, 1) * 10
        )
        features["achievement_quality_score"] = quality_score

        # Specificity (how detailed)
        features["achievement_specificity"] = min(features["achievement_avg_length"] / 10, 10.0)

        return features

    def _extract_behavioral_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract behavioral indicators."""
        features = {}

        experiences = parsed_data.experiences

        if not experiences:
            return {
                "behavior_job_stability": 5.0,  # Neutral
                "behavior_growth_mindset": 0.0,
                "behavior_initiative": 0.0,
                "behavior_collaboration": 0.0,
                "behavior_continuous_learning": 0.0,
                "behavior_problem_solving": 0.0,
                "behavior_ownership": 0.0,
            }

        # Job stability (average tenure)
        tenures = []
        for exp in experiences:
            if exp.start_date and (exp.end_date or exp.current):
                end = exp.end_date or datetime.now()
                tenure = (end - exp.start_date).days / 365.25
                tenures.append(tenure)

        avg_tenure = np.mean(tenures) if tenures else 2.0
        features["behavior_job_stability"] = min(avg_tenure, 10.0)

        # Growth mindset indicators
        growth_keywords = ["learn", "grew", "developed", "studied", "trained", "certified"]
        all_text = " ".join([
            exp.description or "" for exp in experiences
        ] + [
            r for exp in experiences for r in exp.responsibilities
        ]).lower()

        growth_count = sum(all_text.count(kw) for kw in growth_keywords)
        features["behavior_growth_mindset"] = min(float(growth_count) / len(experiences), 5.0)

        # Initiative indicators
        initiative_keywords = ["initiated", "proposed", "started", "launched", "created", "founded"]
        initiative_count = sum(all_text.count(kw) for kw in initiative_keywords)
        features["behavior_initiative"] = min(float(initiative_count), 5.0)

        # Collaboration indicators
        collab_keywords = ["team", "collaborated", "partnered", "coordinated", "worked with"]
        collab_count = sum(all_text.count(kw) for kw in collab_keywords)
        features["behavior_collaboration"] = min(float(collab_count) / len(experiences), 5.0)

        # Continuous learning (certifications + education)
        learning_score = (
            len(parsed_data.certifications) * 2 +
            len(parsed_data.education) +
            len(parsed_data.projects)
        )
        features["behavior_continuous_learning"] = min(float(learning_score), 10.0)

        # Problem-solving indicators
        problem_keywords = ["solved", "fixed", "debugged", "optimized", "improved", "resolved"]
        problem_count = sum(all_text.count(kw) for kw in problem_keywords)
        features["behavior_problem_solving"] = min(float(problem_count) / len(experiences), 5.0)

        # Ownership indicators
        ownership_keywords = ["owned", "responsible for", "managed", "led", "drove"]
        ownership_count = sum(all_text.count(kw) for kw in ownership_keywords)
        features["behavior_ownership"] = min(float(ownership_count) / len(experiences), 5.0)

        return features

    def _extract_network_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract network and social presence features."""
        features = {}

        contact = parsed_data.contact

        # Social presence
        features["network_has_linkedin"] = 1.0 if contact.linkedin else 0.0
        features["network_has_github"] = 1.0 if contact.github else 0.0
        features["network_has_website"] = 1.0 if contact.website else 0.0

        # Social completeness (out of 3 main platforms)
        social_count = sum([
            1 if contact.linkedin else 0,
            1 if contact.github else 0,
            1 if contact.website else 0,
        ])
        features["network_social_completeness"] = social_count / 3.0

        # Online visibility score
        features["network_visibility_score"] = (
            features["network_has_linkedin"] * 3 +
            features["network_has_github"] * 4 +  # GitHub weighted more for tech
            features["network_has_website"] * 3
        )

        return features

    def _extract_writing_quality_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract writing quality metrics."""
        features = {}

        # Combine all written content
        texts = []
        if parsed_data.summary:
            texts.append(parsed_data.summary)
        for exp in parsed_data.experiences:
            if exp.description:
                texts.append(exp.description)
            texts.extend(exp.responsibilities)
            texts.extend(exp.achievements)

        if not texts:
            return {
                "writing_avg_sentence_length": 0.0,
                "writing_vocabulary_diversity": 0.0,
                "writing_professional_tone": 0.0,
                "writing_clarity_score": 0.0,
                "writing_action_verb_ratio": 0.0,
                "writing_complexity": 0.0,
            }

        all_text = " ".join(texts)
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = all_text.lower().split()

        # Average sentence length
        if sentences:
            avg_sent_length = np.mean([len(s.split()) for s in sentences])
            features["writing_avg_sentence_length"] = min(avg_sent_length, 30.0)
        else:
            features["writing_avg_sentence_length"] = 0.0

        # Vocabulary diversity (unique words / total words)
        if words:
            unique_words = len(set(words))
            features["writing_vocabulary_diversity"] = unique_words / len(words)
        else:
            features["writing_vocabulary_diversity"] = 0.0

        # Professional tone (absence of informal language)
        informal_words = ["gonna", "wanna", "yeah", "cool", "awesome", "stuff"]
        informal_count = sum(all_text.lower().count(word) for word in informal_words)
        features["writing_professional_tone"] = max(0, 10 - informal_count) / 10

        # Clarity score (short sentences + common words)
        clarity = (1 - min(features["writing_avg_sentence_length"] / 25, 1)) * 0.5 + \
                 features["writing_professional_tone"] * 0.5
        features["writing_clarity_score"] = clarity

        # Action verb ratio
        action_verbs = ["led", "managed", "created", "developed", "implemented", "designed",
                       "built", "launched", "improved", "optimized", "increased", "reduced"]
        action_count = sum(all_text.lower().count(verb) for verb in action_verbs)
        features["writing_action_verb_ratio"] = min(action_count / len(sentences), 1.0) if sentences else 0.0

        # Complexity (Flesch-Kincaid grade level approximation)
        if words and sentences:
            avg_syllables = 1.5  # Approximation
            complexity = 0.39 * (len(words) / len(sentences)) + 11.8 * avg_syllables - 15.59
            features["writing_complexity"] = max(0, min(complexity, 20.0))
        else:
            features["writing_complexity"] = 0.0

        return features

    def _extract_career_trajectory_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract career trajectory and progression features."""
        features = {}

        experiences = parsed_data.experiences

        if len(experiences) < 2:
            return {
                "trajectory_acceleration": 0.0,
                "trajectory_consistency": 1.0,
                "trajectory_upward_mobility": 0.0,
                "trajectory_company_size_trend": 0.0,
                "trajectory_industry_switches": 0.0,
                "trajectory_role_diversity": 0.0,
                "trajectory_promotion_rate": 0.0,
                "trajectory_career_focus": 1.0,
            }

        # Sort by date
        sorted_exps = sorted(experiences, key=lambda x: x.start_date if x.start_date else datetime.min)

        # Acceleration (is tenure decreasing or increasing?)
        tenures = []
        for exp in sorted_exps:
            if exp.start_date and (exp.end_date or exp.current):
                end = exp.end_date or datetime.now()
                tenure = (end - exp.start_date).days / 365.25
                tenures.append(tenure)

        if len(tenures) >= 2:
            # Positive = getting more stable, Negative = job hopping accelerating
            acceleration = (tenures[-1] - tenures[0]) / len(tenures)
            features["trajectory_acceleration"] = max(-5.0, min(5.0, acceleration))
        else:
            features["trajectory_acceleration"] = 0.0

        # Consistency (std of tenures)
        if tenures:
            consistency = 1 / (1 + np.std(tenures))
            features["trajectory_consistency"] = consistency
        else:
            features["trajectory_consistency"] = 1.0

        # Upward mobility (title changes)
        title_levels = []
        for exp in sorted_exps:
            title = exp.title.lower() if exp.title else ""
            if "senior" in title or "lead" in title or "principal" in title:
                level = 3
            elif "manager" in title or "director" in title:
                level = 4
            elif "vp" in title or "chief" in title or "head" in title:
                level = 5
            elif "junior" in title or "associate" in title:
                level = 1
            else:
                level = 2
            title_levels.append(level)

        if len(title_levels) >= 2:
            mobility = (title_levels[-1] - title_levels[0]) / len(title_levels)
            features["trajectory_upward_mobility"] = max(-2.0, min(2.0, mobility))
        else:
            features["trajectory_upward_mobility"] = 0.0

        # Company size trend (startup -> enterprise or vice versa)
        # Using company name as proxy (simplified)
        large_companies = ["google", "microsoft", "amazon", "apple", "meta", "facebook",
                          "ibm", "oracle", "cisco", "intel", "salesforce"]
        company_sizes = []
        for exp in sorted_exps:
            company = exp.company.lower() if exp.company else ""
            is_large = any(lc in company for lc in large_companies)
            company_sizes.append(1.0 if is_large else 0.5)

        if len(company_sizes) >= 2:
            size_trend = company_sizes[-1] - company_sizes[0]
            features["trajectory_company_size_trend"] = size_trend
        else:
            features["trajectory_company_size_trend"] = 0.0

        # Industry switches
        # This is simplified - in reality would need better industry detection
        features["trajectory_industry_switches"] = 0.0  # Placeholder

        # Role diversity (IC, manager, architect, etc.)
        roles = set()
        for exp in sorted_exps:
            title = exp.title.lower() if exp.title else ""
            for role_type, keywords in self.ROLE_TYPES.items():
                if any(kw in title for kw in keywords):
                    roles.add(role_type)

        features["trajectory_role_diversity"] = float(len(roles))

        # Promotion rate (title level increases per year)
        if len(title_levels) >= 2 and parsed_data.total_experience_years:
            level_change = title_levels[-1] - title_levels[0]
            years = parsed_data.total_experience_years
            promotion_rate = level_change / years if years > 0 else 0
            features["trajectory_promotion_rate"] = max(0, min(promotion_rate, 1.0))
        else:
            features["trajectory_promotion_rate"] = 0.0

        # Career focus (stayed in same domain vs switched)
        # Using role diversity as inverse proxy
        features["trajectory_career_focus"] = 1 / (1 + features["trajectory_role_diversity"] / 3)

        return features

    def _extract_specialization_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract specialization and expertise depth features."""
        features = {}

        skills = parsed_data.skills
        technical_skills = parsed_data.technical_skills

        # Technical depth indicators
        all_text = " ".join([
            parsed_data.summary or "",
            " ".join([exp.description or "" for exp in parsed_data.experiences]),
        ]).lower()

        for depth_area, keywords in self.TECH_DEPTH_KEYWORDS.items():
            count = sum(all_text.count(kw) for kw in keywords)
            features[f"specialization_{depth_area}_depth"] = min(float(count), 5.0)

        # Overall technical depth
        total_depth = sum(features[f"specialization_{area}_depth"]
                         for area in self.TECH_DEPTH_KEYWORDS.keys())
        features["specialization_technical_depth"] = total_depth

        return features

    def _extract_education_quality_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract education quality metrics."""
        features = {}

        education = parsed_data.education

        if not education:
            return {
                "education_top_tier": 0.0,
                "education_gpa_excellence": 0.0,
                "education_research_focus": 0.0,
                "education_honors": 0.0,
                "education_relevance": 0.0,
            }

        # Top-tier universities (simplified list)
        top_universities = ["mit", "stanford", "harvard", "berkeley", "cmu", "caltech",
                           "technion", "oxford", "cambridge", "eth", "epfl"]

        top_tier_count = sum(1 for edu in education
                            if any(uni in edu.institution.lower() for uni in top_universities))
        features["education_top_tier"] = min(float(top_tier_count), 3.0)

        # GPA excellence
        gpas = [edu.gpa for edu in education if edu.gpa]
        if gpas:
            max_gpa = max(gpas)
            features["education_gpa_excellence"] = max(0, (max_gpa - 3.0) * 10)  # Scale from 3.0
        else:
            features["education_gpa_excellence"] = 0.0

        # Research focus (PhD, publications)
        has_phd = any(edu.level == EducationLevel.DOCTORATE for edu in education)
        features["education_research_focus"] = (
            (1.0 if has_phd else 0.0) * 5 +
            min(len(parsed_data.publications), 3) * 2
        )

        # Honors (magna cum laude, dean's list, etc.)
        # This would require parsing degree strings - placeholder
        features["education_honors"] = 0.0

        # Relevance to tech (STEM fields)
        stem_keywords = ["computer", "engineering", "science", "mathematics", "technology", "physics"]
        relevant_count = sum(1 for edu in education
                            if edu.field_of_study and any(kw in edu.field_of_study.lower()
                                                          for kw in stem_keywords))
        features["education_relevance"] = float(relevant_count)

        return features

    def _extract_experience_depth_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """Extract experience depth and breadth features."""
        features = {}

        experiences = parsed_data.experiences

        if not experiences:
            return {
                "experience_avg_responsibilities": 0.0,
                "experience_avg_achievements": 0.0,
                "experience_detail_score": 0.0,
                "experience_breadth": 0.0,
                "experience_impact_scope": 0.0,
            }

        # Average responsibilities per job
        resp_counts = [len(exp.responsibilities) for exp in experiences]
        features["experience_avg_responsibilities"] = np.mean(resp_counts) if resp_counts else 0.0

        # Average achievements per job
        ach_counts = [len(exp.achievements) for exp in experiences]
        features["experience_avg_achievements"] = np.mean(ach_counts) if ach_counts else 0.0

        # Detail score (how well documented)
        detail_scores = []
        for exp in experiences:
            score = (
                (1 if exp.description else 0) * 2 +
                min(len(exp.responsibilities), 5) +
                min(len(exp.achievements), 3) * 1.5
            )
            detail_scores.append(score)

        features["experience_detail_score"] = np.mean(detail_scores) if detail_scores else 0.0

        # Breadth (unique companies and roles)
        unique_companies = len(set(exp.company for exp in experiences if exp.company))
        unique_titles = len(set(exp.title for exp in experiences if exp.title))
        features["experience_breadth"] = float(unique_companies + unique_titles)

        # Impact scope (team size mentions, user numbers, etc.)
        # This would require NER - placeholder
        features["experience_impact_scope"] = 0.0

        return features
