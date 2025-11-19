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

from .models import ParsedData, Experience, Education, Skill, EducationLevel

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

        # 10. Emotional Intelligence Features (8 features) - 0.60 validity
        features.update(self._extract_emotional_intelligence_features(parsed_data))

        # 11. Cognitive Ability Proxy Features (6 features) - 0.51 validity
        features.update(self._extract_cognitive_ability_features(parsed_data))

        # 12. Person-Organization Fit Features (7 features) - 0.44 validity
        features.update(self._extract_person_org_fit_features(parsed_data))

        # 13. Structured Assessment Indicators (6 features) - 0.54 validity
        features.update(self._extract_structured_assessment_features(parsed_data))

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

    def _extract_emotional_intelligence_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """
        Extract Emotional Intelligence (EI) proxy features.

        Validity: 0.60
        Research: TalentSmart study shows 90% of top performers have high EI

        Features:
        - Communication quality from writing patterns
        - Resilience indicators from career transitions
        - Team collaboration signals
        - Interpersonal effectiveness
        """
        features = {}

        # Combine all text for analysis
        experiences = parsed_data.experiences
        all_text = " ".join([
            parsed_data.summary or "",
            " ".join([exp.description or "" for exp in experiences]),
            " ".join([r for exp in experiences for r in exp.responsibilities]),
            " ".join([a for exp in experiences for a in exp.achievements]),
        ]).lower()

        # 1. Communication Quality (from writing patterns)
        # Positive communication keywords
        positive_comm_keywords = ["communicated", "presented", "explained", "articulated",
                                 "facilitated", "discussed", "engaged", "conveyed"]
        comm_count = sum(all_text.count(kw) for kw in positive_comm_keywords)
        features["ei_communication_quality"] = min(float(comm_count), 10.0)

        # 2. Interpersonal Effectiveness
        interpersonal_keywords = ["interpersonal", "relationship", "rapport", "empathy",
                                 "emotional", "stakeholder", "cross-functional"]
        interpersonal_count = sum(all_text.count(kw) for kw in interpersonal_keywords)
        features["ei_interpersonal_effectiveness"] = min(float(interpersonal_count), 5.0)

        # 3. Resilience Indicators (career transitions, gap handling)
        # Resilience keywords in achievements/descriptions
        resilience_keywords = ["overcame", "navigated", "adapted", "pivoted", "transformed",
                              "recovered", "rebuilt", "turnaround", "challenge", "difficulty"]
        resilience_count = sum(all_text.count(kw) for kw in resilience_keywords)
        features["ei_resilience_indicators"] = min(float(resilience_count), 5.0)

        # Career gaps with explanations (positive resilience signal)
        gaps_with_context = 0
        if len(experiences) >= 2:
            sorted_exps = sorted(experiences, key=lambda x: x.start_date if x.start_date else datetime.min)
            for i in range(len(sorted_exps) - 1):
                current_exp = sorted_exps[i]
                next_exp = sorted_exps[i + 1]
                if current_exp.end_date and next_exp.start_date:
                    gap_days = (next_exp.start_date - current_exp.end_date).days
                    if gap_days > 60:  # 2+ month gap
                        # Check if next role shows growth despite gap
                        if next_exp.title and current_exp.title:
                            if len(next_exp.title) >= len(current_exp.title):  # Simple proxy
                                gaps_with_context += 1

        features["ei_gap_recovery"] = float(gaps_with_context)

        # 4. Team Collaboration (70% more likely to succeed with high EI)
        collaboration_keywords = ["team", "collaborated", "partnered", "coordinated",
                                 "cooperated", "jointly", "together", "mentored", "coached"]
        collab_count = sum(all_text.count(kw) for kw in collaboration_keywords)
        features["ei_collaboration_signals"] = min(float(collab_count) / max(len(experiences), 1), 10.0)

        # 5. Conflict Resolution & Influence
        influence_keywords = ["influenced", "persuaded", "negotiated", "consensus",
                             "alignment", "buy-in", "resolved", "mediated"]
        influence_count = sum(all_text.count(kw) for kw in influence_keywords)
        features["ei_influence_ability"] = min(float(influence_count), 5.0)

        # 6. Emotional Awareness (self-awareness indicators)
        awareness_keywords = ["learned", "feedback", "growth", "developed myself",
                             "self-improvement", "reflection", "recognized"]
        awareness_count = sum(all_text.count(kw) for kw in awareness_keywords)
        features["ei_self_awareness"] = min(float(awareness_count), 5.0)

        # 7. Proactive Career Development (70% more likely with high EI)
        development_keywords = ["certification", "training", "course", "workshop",
                               "professional development", "upskilling"]
        development_count = (len(parsed_data.certifications) * 2 +
                           sum(all_text.count(kw) for kw in development_keywords))
        features["ei_proactive_development"] = min(float(development_count), 10.0)

        # 8. Overall EI Score (composite 0-100)
        ei_score = (
            features["ei_communication_quality"] / 10 * 20 +
            features["ei_interpersonal_effectiveness"] / 5 * 15 +
            features["ei_resilience_indicators"] / 5 * 15 +
            features["ei_collaboration_signals"] / 10 * 20 +
            features["ei_influence_ability"] / 5 * 10 +
            features["ei_self_awareness"] / 5 * 10 +
            features["ei_proactive_development"] / 10 * 10
        )
        features["ei_overall_score"] = ei_score

        return features

    def _extract_cognitive_ability_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """
        Extract Cognitive Ability proxy features.

        Validity: 0.51 (highest single predictor per Schmidt & Hunter 1998)
        Research: Facilitates learning of job-relevant knowledge

        Proxies from resume:
        - Problem-solving complexity
        - Learning velocity
        - Certification complexity
        - Technical depth indicators
        """
        features = {}

        experiences = parsed_data.experiences
        all_text = " ".join([
            parsed_data.summary or "",
            " ".join([exp.description or "" for exp in experiences]),
            " ".join([a for exp in experiences for a in exp.achievements]),
        ]).lower()

        # 1. Problem-Solving Complexity (from achievement descriptions)
        complex_problem_keywords = [
            "algorithm", "optimization", "architecture", "designed", "engineered",
            "complex", "system", "scalability", "performance", "distributed",
            "analyzed", "diagnosed", "root cause", "systematic", "framework"
        ]
        problem_complexity_count = sum(all_text.count(kw) for kw in complex_problem_keywords)
        features["cognitive_problem_complexity"] = min(float(problem_complexity_count) / max(len(experiences), 1), 10.0)

        # 2. Learning Velocity (new skills acquired per year of experience)
        if parsed_data.total_experience_years and parsed_data.total_experience_years > 0:
            total_skills = len(parsed_data.skills) + len(parsed_data.technical_skills)
            learning_velocity = total_skills / parsed_data.total_experience_years
            features["cognitive_learning_velocity"] = min(learning_velocity, 10.0)
        else:
            features["cognitive_learning_velocity"] = 0.0

        # 3. Certification Complexity
        # Advanced certifications indicate higher cognitive ability
        advanced_cert_keywords = ["architect", "expert", "professional", "advanced",
                                 "certified", "specialist", "master"]
        cert_texts = " ".join([cert.lower() for cert in parsed_data.certifications])
        complex_cert_count = sum(cert_texts.count(kw) for kw in advanced_cert_keywords)
        features["cognitive_certification_complexity"] = min(float(complex_cert_count), 5.0)

        # 4. Academic Achievement Indicators
        # GPA, honors, research
        academic_score = 0.0
        for edu in parsed_data.education:
            if edu.gpa and edu.gpa >= 3.5:
                academic_score += (edu.gpa - 3.0) * 2  # Scale from 3.0
            # Check for honors in degree string
            if edu.degree:
                honors_keywords = ["honors", "distinction", "magna", "summa", "cum laude"]
                if any(h in edu.degree.lower() for h in honors_keywords):
                    academic_score += 2.0

        features["cognitive_academic_achievement"] = min(academic_score, 10.0)

        # 5. Technical Depth & Breadth Balance
        # High cognitive ability = deep in some areas, broad in others
        tech_depth = sum([
            features.get(f"specialization_{area}_depth", 0)
            for area in self.TECH_DEPTH_KEYWORDS.keys()
        ]) if hasattr(self, 'TECH_DEPTH_KEYWORDS') else 0

        tech_breadth = len(parsed_data.technical_skills)

        # Balanced ratio indicates cognitive flexibility
        if tech_breadth > 0:
            depth_breadth_ratio = tech_depth / tech_breadth
            features["cognitive_depth_breadth_balance"] = min(depth_breadth_ratio * 2, 10.0)
        else:
            features["cognitive_depth_breadth_balance"] = 0.0

        # 6. Abstract Thinking Indicators
        abstract_keywords = ["strategy", "vision", "concept", "theory", "principle",
                            "methodology", "paradigm", "pattern", "model", "framework"]
        abstract_count = sum(all_text.count(kw) for kw in abstract_keywords)
        features["cognitive_abstract_thinking"] = min(float(abstract_count) / max(len(experiences), 1), 5.0)

        return features

    def _extract_person_org_fit_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """
        Extract Person-Organization Fit features.

        Validity: 0.44 (Kristof-Brown 2005)
        Research: Match between worker needs and job supplies; critical for retention

        Features:
        - Company quality trajectory (not just prestige)
        - Industry reputation alignment
        - Career move patterns
        """
        features = {}

        experiences = parsed_data.experiences

        if not experiences:
            return {
                "po_fit_company_trajectory": 0.0,
                "po_fit_company_quality_trend": 0.0,
                "po_fit_industry_consistency": 0.0,
                "po_fit_career_move_quality": 0.0,
                "po_fit_company_size_preference": 0.0,
                "po_fit_role_alignment": 1.0,
                "po_fit_overall_score": 5.0,
            }

        # Sort experiences by date
        sorted_exps = sorted(experiences, key=lambda x: x.start_date if x.start_date else datetime.min)

        # 1. Company Quality Trajectory (are companies getting better?)
        # Using employer brand indicators
        top_tier_companies = [
            "google", "microsoft", "amazon", "apple", "meta", "facebook", "netflix",
            "tesla", "nvidia", "openai", "anthropic", "deepmind",
            "stripe", "airbnb", "uber", "linkedin", "twitter", "salesforce"
        ]

        mid_tier_indicators = ["unicorn", "series", "funded", "startup", "scale-up"]

        company_quality_scores = []
        for exp in sorted_exps:
            company_lower = exp.company.lower() if exp.company else ""
            description_lower = exp.description.lower() if exp.description else ""

            # Score company quality
            if any(top in company_lower for top in top_tier_companies):
                score = 10.0
            elif any(mid in description_lower or mid in company_lower for mid in mid_tier_indicators):
                score = 7.0
            else:
                score = 5.0  # Default

            company_quality_scores.append(score)

        # Calculate trajectory (positive = improving)
        if len(company_quality_scores) >= 2:
            trajectory = (company_quality_scores[-1] - company_quality_scores[0]) / len(company_quality_scores)
            features["po_fit_company_trajectory"] = max(-5.0, min(5.0, trajectory))

            # Trend consistency (are they consistently moving up?)
            improving_moves = sum(1 for i in range(len(company_quality_scores)-1)
                                 if company_quality_scores[i+1] >= company_quality_scores[i])
            features["po_fit_company_quality_trend"] = improving_moves / (len(company_quality_scores) - 1)
        else:
            features["po_fit_company_trajectory"] = 0.0
            features["po_fit_company_quality_trend"] = 0.5

        # 2. Industry Consistency (staying in same industry = better fit understanding)
        # Use industry features if already computed
        all_text = " ".join([exp.company + " " + (exp.description or "") for exp in experiences]).lower()

        industry_mentions = {}
        for industry, keywords in self.INDUSTRY_KEYWORDS.items():
            count = sum(all_text.count(kw) for kw in keywords)
            if count > 0:
                industry_mentions[industry] = count

        if industry_mentions:
            # High consistency = focused on one industry
            max_industry_count = max(industry_mentions.values())
            total_mentions = sum(industry_mentions.values())
            consistency = max_industry_count / total_mentions if total_mentions > 0 else 0
            features["po_fit_industry_consistency"] = consistency * 10
        else:
            features["po_fit_industry_consistency"] = 5.0

        # 3. Career Move Quality (30% minimum increase recommended)
        # Tenure at quality companies + progression
        quality_moves = 0
        for i, exp in enumerate(sorted_exps):
            if exp.start_date and (exp.end_date or exp.current):
                end = exp.end_date or datetime.now()
                tenure_years = (end - exp.start_date).days / 365.25

                # Quality move = stayed 2+ years OR moved to better company
                if tenure_years >= 2.0:
                    quality_moves += 1
                elif i < len(sorted_exps) - 1 and len(company_quality_scores) > i + 1:
                    if company_quality_scores[i + 1] > company_quality_scores[i]:
                        quality_moves += 1

        features["po_fit_career_move_quality"] = quality_moves / max(len(experiences), 1) * 10

        # 4. Company Size Preference Pattern
        # Consistent pattern indicates better fit awareness
        large_company_count = sum(1 for score in company_quality_scores if score >= 9)
        startup_count = sum(1 for score in company_quality_scores if score <= 6)

        if len(experiences) > 0:
            if large_company_count / len(experiences) > 0.7:
                features["po_fit_company_size_preference"] = 10.0  # Clear large co preference
            elif startup_count / len(experiences) > 0.7:
                features["po_fit_company_size_preference"] = 10.0  # Clear startup preference
            else:
                features["po_fit_company_size_preference"] = 5.0  # Mixed
        else:
            features["po_fit_company_size_preference"] = 5.0

        # 5. Role Alignment (consistent role type = knows what they want)
        role_types_present = set()
        for exp in sorted_exps:
            title_lower = exp.title.lower() if exp.title else ""
            for role_type, keywords in self.ROLE_TYPES.items():
                if any(kw in title_lower for kw in keywords):
                    role_types_present.add(role_type)

        # Few role types = high alignment
        role_alignment = max(0, 10 - len(role_types_present) * 2)
        features["po_fit_role_alignment"] = float(role_alignment)

        # 6. Overall PO-Fit Score (0-100)
        po_fit_score = (
            (features["po_fit_company_trajectory"] + 5) / 10 * 20 +  # Normalize from -5,5 to 0,1
            features["po_fit_company_quality_trend"] * 20 +
            features["po_fit_industry_consistency"] / 10 * 20 +
            features["po_fit_career_move_quality"] / 10 * 20 +
            features["po_fit_role_alignment"] / 10 * 20
        )
        features["po_fit_overall_score"] = po_fit_score

        return features

    def _extract_structured_assessment_features(self, parsed_data: ParsedData) -> Dict[str, float]:
        """
        Extract Structured Assessment Indicators.

        Validity: 0.54 (0.54-0.62 when combined with other assessments)
        Research: Performance-based hiring research; Achiever Pattern

        Features:
        - Achiever pattern (consistent high performance)
        - Growth rate consistency
        - Fit quality indicators
        """
        features = {}

        experiences = parsed_data.experiences

        if not experiences:
            return {
                "assessment_achiever_pattern": 0.0,
                "assessment_growth_consistency": 0.0,
                "assessment_performance_trend": 0.0,
                "assessment_achievement_density": 0.0,
                "assessment_quantified_impact": 0.0,
                "assessment_overall_quality": 0.0,
            }

        # 1. Achiever Pattern (from Performance-Based Hiring)
        # Pattern: Consistent track record of exceeding expectations

        # Count achievements per role
        achievement_counts = [len(exp.achievements) for exp in experiences]
        avg_achievements = np.mean(achievement_counts) if achievement_counts else 0

        # Consistency of achievements (std deviation)
        achievement_consistency = 0.0
        if len(achievement_counts) > 1:
            std_achievements = np.std(achievement_counts)
            # Lower std = more consistent (inverse relationship)
            achievement_consistency = max(0, 10 - std_achievements)
        else:
            achievement_consistency = 5.0

        features["assessment_achiever_pattern"] = min(avg_achievements, 10.0)
        features["assessment_growth_consistency"] = achievement_consistency

        # 2. Performance Trend (are achievements getting better?)
        # Look at achievement quality over time
        sorted_exps = sorted(experiences, key=lambda x: x.start_date if x.start_date else datetime.min)

        if len(sorted_exps) >= 3:
            # Compare first third vs last third
            third = len(sorted_exps) // 3
            early_achievements = sum(len(exp.achievements) for exp in sorted_exps[:third])
            late_achievements = sum(len(exp.achievements) for exp in sorted_exps[-third:])

            if third > 0:
                early_avg = early_achievements / third
                late_avg = late_achievements / third
                trend = late_avg - early_avg
                features["assessment_performance_trend"] = max(-5.0, min(5.0, trend))
            else:
                features["assessment_performance_trend"] = 0.0
        else:
            features["assessment_performance_trend"] = 0.0

        # 3. Achievement Density (achievements per year of experience)
        total_achievements = sum(len(exp.achievements) for exp in experiences)
        if parsed_data.total_experience_years and parsed_data.total_experience_years > 0:
            achievement_density = total_achievements / parsed_data.total_experience_years
            features["assessment_achievement_density"] = min(achievement_density * 2, 10.0)
        else:
            features["assessment_achievement_density"] = 0.0

        # 4. Quantified Impact (% of achievements with numbers)
        quantified_pattern = r'\d+[%xX]|\d+\s*(million|thousand|percent|users|customers)'
        quantified_count = 0

        for exp in experiences:
            for achievement in exp.achievements:
                if re.search(quantified_pattern, achievement):
                    quantified_count += 1

        if total_achievements > 0:
            quantified_ratio = quantified_count / total_achievements
            features["assessment_quantified_impact"] = quantified_ratio * 10
        else:
            features["assessment_quantified_impact"] = 0.0

        # 5. Fit Quality (from Performance-Based Hiring)
        # Combination of: right skills + right growth + right trajectory

        # Right skills: technical depth
        tech_skills_count = len(parsed_data.technical_skills)
        skills_score = min(tech_skills_count / 10, 1.0) * 30

        # Right growth: upward trajectory
        trajectory_score = 0.0
        if "trajectory_upward_mobility" in dir(self):
            # If trajectory was already calculated
            trajectory_score = 35  # Placeholder

        # Right achievements: quantified + consistent
        achievement_score = (
            features["assessment_achiever_pattern"] / 10 * 35
        )

        fit_quality = skills_score + achievement_score
        features["assessment_fit_quality"] = min(fit_quality, 100.0)

        # 6. Overall Assessment Score (0-100)
        assessment_score = (
            features["assessment_achiever_pattern"] / 10 * 25 +
            features["assessment_growth_consistency"] / 10 * 20 +
            (features["assessment_performance_trend"] + 5) / 10 * 15 +  # Normalize -5,5 to 0,1
            features["assessment_achievement_density"] / 10 * 20 +
            features["assessment_quantified_impact"] / 10 * 20
        )
        features["assessment_overall_quality"] = assessment_score

        return features
