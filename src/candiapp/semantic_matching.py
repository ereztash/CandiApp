"""
Semantic Skill Matching Module

Uses embeddings and similarity metrics to match skills semantically:
- Word embeddings (Word2Vec, GloVe)
- Sentence embeddings (for skill descriptions)
- Semantic similarity calculation
- Fuzzy skill matching
- Skill taxonomy and relationships
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import warnings

logger = logging.getLogger(__name__)


class SemanticSkillMatcher:
    """
    Semantic skill matcher using embeddings and similarity metrics.

    Provides:
    - Semantic skill similarity
    - Skill synonym detection
    - Skill category clustering
    - Fuzzy skill matching
    """

    # Skill taxonomy - related skills
    SKILL_TAXONOMY = {
        # Programming languages
        "python": {"python3", "python2", "py", "cpython", "jython", "pypy"},
        "javascript": {"js", "ecmascript", "es6", "es2015", "node.js", "nodejs"},
        "java": {"java8", "java11", "java17", "jdk", "jvm"},
        "c++": {"cpp", "c plus plus", "cxx"},
        "c#": {"csharp", "c sharp", "dotnet", ".net"},

        # Web frameworks
        "react": {"reactjs", "react.js", "react native"},
        "angular": {"angularjs", "angular2", "angular.js"},
        "vue": {"vuejs", "vue.js"},
        "django": {"python django"},
        "flask": {"python flask"},
        "spring": {"spring boot", "spring framework"},

        # Databases
        "postgresql": {"postgres", "psql"},
        "mysql": {"mariadb"},
        "mongodb": {"mongo", "document database"},
        "redis": {"redis cache"},

        # Cloud platforms
        "aws": {"amazon web services", "amazon aws", "ec2", "s3"},
        "azure": {"microsoft azure", "azure cloud"},
        "gcp": {"google cloud", "google cloud platform"},

        # DevOps
        "docker": {"containerization", "container"},
        "kubernetes": {"k8s", "container orchestration"},
        "jenkins": {"ci/cd", "continuous integration"},
        "git": {"version control", "github", "gitlab", "bitbucket"},

        # Data Science
        "machine learning": {"ml", "supervised learning", "unsupervised learning"},
        "deep learning": {"dl", "neural networks", "cnn", "rnn"},
        "tensorflow": {"tf", "tensorflow 2"},
        "pytorch": {"torch"},
        "scikit-learn": {"sklearn", "scikit learn"},

        # Mobile
        "android": {"android development", "android studio"},
        "ios": {"swift", "objective-c", "xcode"},
        "react native": {"rn", "react-native"},
        "flutter": {"dart flutter"},
    }

    # Skill categories for semantic grouping
    SKILL_CATEGORIES = {
        "programming_languages": [
            "python", "java", "javascript", "c++", "c#", "go", "rust", "ruby",
            "php", "swift", "kotlin", "typescript", "scala", "r"
        ],
        "web_frontend": [
            "react", "angular", "vue", "html", "css", "sass", "webpack",
            "redux", "next.js", "gatsby"
        ],
        "web_backend": [
            "node.js", "django", "flask", "spring", "express", "fastapi",
            "asp.net", "rails", "laravel"
        ],
        "databases": [
            "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
            "cassandra", "dynamodb", "neo4j"
        ],
        "cloud": [
            "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
            "ansible", "cloudformation"
        ],
        "data_science": [
            "machine learning", "deep learning", "tensorflow", "pytorch",
            "pandas", "numpy", "scikit-learn", "keras", "nlp"
        ],
        "mobile": [
            "android", "ios", "react native", "flutter", "xamarin", "ionic"
        ],
        "devops": [
            "ci/cd", "jenkins", "gitlab", "docker", "kubernetes", "terraform",
            "ansible", "monitoring", "prometheus"
        ],
    }

    # Skill similarity matrix (pre-computed for common pairs)
    # In production, would use actual embeddings
    SKILL_SIMILARITIES = {
        ("python", "java"): 0.65,
        ("python", "ruby"): 0.70,
        ("javascript", "typescript"): 0.85,
        ("react", "angular"): 0.75,
        ("react", "vue"): 0.80,
        ("postgresql", "mysql"): 0.85,
        ("aws", "azure"): 0.75,
        ("aws", "gcp"): 0.75,
        ("docker", "kubernetes"): 0.80,
        ("tensorflow", "pytorch"): 0.85,
        ("android", "ios"): 0.60,
    }

    def __init__(self, use_embeddings: bool = False):
        """
        Initialize semantic skill matcher.

        Args:
            use_embeddings: Use actual word embeddings (requires gensim/sentence-transformers)
        """
        self.use_embeddings = use_embeddings
        self.embedding_model = None

        if use_embeddings:
            self._load_embedding_model()

    def _load_embedding_model(self):
        """Load pre-trained embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded sentence embedding model")
        except ImportError:
            logger.warning("⚠️  sentence-transformers not installed")
            logger.warning("   Install with: pip install sentence-transformers")
            logger.warning("   Falling back to rule-based matching")
            self.use_embeddings = False

    def calculate_skill_similarity(self, skill1: str, skill2: str) -> float:
        """
        Calculate semantic similarity between two skills.

        Args:
            skill1: First skill
            skill2: Second skill

        Returns:
            Similarity score (0-1)
        """
        skill1_lower = skill1.lower().strip()
        skill2_lower = skill2.lower().strip()

        # Exact match
        if skill1_lower == skill2_lower:
            return 1.0

        # Check taxonomy
        if self._are_related_in_taxonomy(skill1_lower, skill2_lower):
            return 0.9

        # Check pre-computed similarities
        pair = tuple(sorted([skill1_lower, skill2_lower]))
        if pair in self.SKILL_SIMILARITIES:
            return self.SKILL_SIMILARITIES[pair]

        # Same category
        category_similarity = self._calculate_category_similarity(skill1_lower, skill2_lower)
        if category_similarity > 0:
            return category_similarity

        # Use embeddings if available
        if self.use_embeddings and self.embedding_model:
            return self._calculate_embedding_similarity(skill1, skill2)

        # Fuzzy string matching
        return self._calculate_fuzzy_similarity(skill1_lower, skill2_lower)

    def _are_related_in_taxonomy(self, skill1: str, skill2: str) -> bool:
        """Check if skills are related in taxonomy."""
        for primary_skill, related_skills in self.SKILL_TAXONOMY.items():
            if skill1 == primary_skill or skill1 in related_skills:
                if skill2 == primary_skill or skill2 in related_skills:
                    return True
        return False

    def _calculate_category_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity based on category membership."""
        categories1 = set()
        categories2 = set()

        for category, skills in self.SKILL_CATEGORIES.items():
            if skill1 in skills:
                categories1.add(category)
            if skill2 in skills:
                categories2.add(category)

        if not categories1 or not categories2:
            return 0.0

        # Jaccard similarity of categories
        intersection = len(categories1 & categories2)
        union = len(categories1 | categories2)

        if union == 0:
            return 0.0

        return (intersection / union) * 0.7  # Scale down since it's category-level

    def _calculate_embedding_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity using embeddings."""
        if not self.embedding_model:
            return 0.0

        try:
            # Encode skills
            embeddings = self.embedding_model.encode([skill1, skill2])

            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating embedding similarity: {e}")
            return 0.0

    def _calculate_fuzzy_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate fuzzy string similarity."""
        # Levenshtein distance approximation
        len1, len2 = len(skill1), len(skill2)

        if len1 == 0 or len2 == 0:
            return 0.0

        # Simple character overlap
        set1, set2 = set(skill1), set(skill2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        char_similarity = intersection / union if union > 0 else 0.0

        # Substring matching
        substring_score = 0.0
        if skill1 in skill2 or skill2 in skill1:
            substring_score = 0.3

        return max(char_similarity, substring_score)

    def find_similar_skills(
        self,
        skill: str,
        skill_database: List[str],
        threshold: float = 0.7,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find skills similar to a given skill.

        Args:
            skill: Target skill
            skill_database: List of skills to search
            threshold: Minimum similarity threshold
            top_k: Maximum number of results

        Returns:
            List of (skill, similarity_score) tuples
        """
        similarities = []

        for db_skill in skill_database:
            similarity = self.calculate_skill_similarity(skill, db_skill)
            if similarity >= threshold:
                similarities.append((db_skill, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def match_required_skills(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        min_similarity: float = 0.75
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Match candidate skills to required skills semantically.

        Args:
            candidate_skills: Skills the candidate has
            required_skills: Skills required for the job
            min_similarity: Minimum similarity to consider a match

        Returns:
            Dictionary mapping required skills to matching candidate skills
        """
        matches = {}

        for required_skill in required_skills:
            matching_skills = []

            for candidate_skill in candidate_skills:
                similarity = self.calculate_skill_similarity(required_skill, candidate_skill)

                if similarity >= min_similarity:
                    matching_skills.append((candidate_skill, similarity))

            # Sort by similarity
            matching_skills.sort(key=lambda x: x[1], reverse=True)

            matches[required_skill] = matching_skills

        return matches

    def calculate_skill_coverage(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        min_similarity: float = 0.75
    ) -> Dict[str, float]:
        """
        Calculate skill coverage metrics.

        Args:
            candidate_skills: Skills the candidate has
            required_skills: Skills required for the job
            min_similarity: Minimum similarity to consider a match

        Returns:
            Dictionary of coverage metrics
        """
        matches = self.match_required_skills(candidate_skills, required_skills, min_similarity)

        # Count matched skills
        matched_count = sum(1 for skill_matches in matches.values() if skill_matches)

        # Calculate coverage percentage
        coverage_pct = (matched_count / len(required_skills) * 100) if required_skills else 0

        # Calculate average match quality
        all_similarities = [
            similarity
            for skill_matches in matches.values()
            for _, similarity in skill_matches
        ]
        avg_match_quality = np.mean(all_similarities) if all_similarities else 0.0

        # Identify missing skills
        missing_skills = [skill for skill, skill_matches in matches.items() if not skill_matches]

        # Identify exact matches vs partial matches
        exact_matches = sum(1 for skill_matches in matches.values()
                          if skill_matches and skill_matches[0][1] >= 0.95)
        partial_matches = matched_count - exact_matches

        return {
            "coverage_pct": coverage_pct,
            "matched_count": matched_count,
            "missing_count": len(missing_skills),
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "avg_match_quality": avg_match_quality,
            "missing_skills": missing_skills,
        }

    def expand_skill_query(self, skill: str, max_expansions: int = 5) -> List[str]:
        """
        Expand a skill query with related skills.

        Useful for job search.

        Args:
            skill: Skill to expand
            max_expansions: Maximum number of related skills to return

        Returns:
            List of related skills
        """
        related = []

        skill_lower = skill.lower()

        # Check taxonomy
        if skill_lower in self.SKILL_TAXONOMY:
            related.extend(list(self.SKILL_TAXONOMY[skill_lower]))

        # Find in taxonomy as related skill
        for primary, related_skills in self.SKILL_TAXONOMY.items():
            if skill_lower in related_skills:
                related.append(primary)
                related.extend([s for s in related_skills if s != skill_lower])

        # Add skills from same category
        for category, skills in self.SKILL_CATEGORIES.items():
            if skill_lower in skills:
                related.extend([s for s in skills if s != skill_lower])

        # Deduplicate and limit
        related = list(set(related))[:max_expansions]

        return related

    def cluster_skills(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Cluster skills into categories.

        Args:
            skills: List of skills to cluster

        Returns:
            Dictionary mapping categories to skills
        """
        clusters = defaultdict(list)

        for skill in skills:
            skill_lower = skill.lower()
            assigned = False

            # Assign to categories
            for category, category_skills in self.SKILL_CATEGORIES.items():
                if skill_lower in category_skills:
                    clusters[category].append(skill)
                    assigned = True

            # Check taxonomy
            if not assigned:
                for primary, related_skills in self.SKILL_TAXONOMY.items():
                    if skill_lower == primary or skill_lower in related_skills:
                        # Find which category this primary skill belongs to
                        for category, category_skills in self.SKILL_CATEGORIES.items():
                            if primary in category_skills:
                                clusters[category].append(skill)
                                assigned = True
                                break

            # Uncategorized
            if not assigned:
                clusters["other"].append(skill)

        return dict(clusters)


# Utility functions
def calculate_semantic_skill_match_score(
    candidate_skills: List[str],
    job_skills: List[str],
    matcher: Optional[SemanticSkillMatcher] = None
) -> float:
    """
    Calculate overall semantic skill match score.

    Args:
        candidate_skills: Candidate's skills
        job_skills: Job requirements
        matcher: Skill matcher instance (creates new if None)

    Returns:
        Match score (0-100)
    """
    if matcher is None:
        matcher = SemanticSkillMatcher()

    coverage = matcher.calculate_skill_coverage(candidate_skills, job_skills)

    # Weighted score
    score = (
        coverage["coverage_pct"] * 0.6 +  # Coverage is most important
        coverage["exact_matches"] / max(len(job_skills), 1) * 100 * 0.3 +  # Exact matches
        coverage["avg_match_quality"] * 100 * 0.1  # Match quality
    )

    return min(score, 100.0)


# Import numpy
try:
    import numpy as np
except ImportError:
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
