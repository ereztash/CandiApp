"""
BERT-based Feature Engineering

Uses pre-trained transformer models (BERT, RoBERTa, etc.) for:
- Contextual embeddings
- Semantic similarity
- Text classification features
- Sentiment analysis
- Resume quality scoring
"""

import logging
from typing import List, Dict, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)


class BERTFeatureExtractor:
    """
    Extract features using BERT and other transformer models.

    Provides:
    - Contextual embeddings for resume sections
    - Semantic similarity between resume and job description
    - Resume quality classification
    - Skill relevance scoring
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        use_gpu: bool = False
    ):
        """
        Initialize BERT feature extractor.

        Args:
            model_name: HuggingFace model name
            use_gpu: Use GPU if available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load BERT model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

            # Move to GPU if requested and available
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info(f"✅ Loaded {self.model_name} on GPU")
            else:
                logger.info(f"✅ Loaded {self.model_name} on CPU")

            self.model.eval()  # Set to evaluation mode

        except ImportError:
            logger.warning("⚠️  transformers library not installed")
            logger.warning("   Install with: pip install transformers torch")
            logger.warning("   BERT features will be disabled")
        except Exception as e:
            logger.error(f"❌ Error loading BERT model: {e}")

    def extract_embedding(self, text: str, pooling: str = "mean") -> Optional[List[float]]:
        """
        Extract BERT embedding for text.

        Args:
            text: Input text
            pooling: Pooling strategy ("mean", "cls", "max")

        Returns:
            Embedding vector (768-dim for BERT-base)
        """
        if not self.model or not self.tokenizer:
            return None

        try:
            import torch

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Move to GPU if using
            if self.use_gpu and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Pool embeddings
            if pooling == "cls":
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            elif pooling == "mean":
                # Mean pooling over all tokens
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
            elif pooling == "max":
                # Max pooling over all tokens
                embedding = outputs.last_hidden_state.max(dim=1)[0].cpu().numpy()[0]
            else:
                raise ValueError(f"Unknown pooling strategy: {pooling}")

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error extracting BERT embedding: {e}")
            return None

    def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate semantic similarity between two texts using BERT.

        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ("cosine", "euclidean")

        Returns:
            Similarity score (0-1 for cosine, lower is better for euclidean)
        """
        if not self.model:
            return 0.0

        # Get embeddings
        emb1 = self.extract_embedding(text1)
        emb2 = self.extract_embedding(text2)

        if emb1 is None or emb2 is None:
            return 0.0

        try:
            import numpy as np

            emb1 = np.array(emb1)
            emb2 = np.array(emb2)

            if metric == "cosine":
                # Cosine similarity
                dot_product = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                similarity = dot_product / (norm1 * norm2)
                return float(similarity)

            elif metric == "euclidean":
                # Euclidean distance
                distance = np.linalg.norm(emb1 - emb2)
                return float(distance)

            else:
                raise ValueError(f"Unknown metric: {metric}")

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def score_resume_quality(self, resume_text: str) -> Dict[str, float]:
        """
        Score resume quality using BERT-based features.

        Args:
            resume_text: Full resume text

        Returns:
            Dictionary of quality scores
        """
        if not self.model:
            return {"overall_quality": 0.5}

        scores = {}

        # Extract sections (simplified)
        sections = self._split_into_sections(resume_text)

        # Score each section
        for section_name, section_text in sections.items():
            if section_text:
                embedding = self.extract_embedding(section_text)
                if embedding:
                    # Use embedding norm as proxy for informativeness
                    import numpy as np
                    norm = np.linalg.norm(embedding)
                    scores[f"{section_name}_quality"] = min(norm / 10, 1.0)

        # Overall quality (average of sections)
        if scores:
            scores["overall_quality"] = sum(scores.values()) / len(scores)
        else:
            scores["overall_quality"] = 0.5

        return scores

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split resume into sections."""
        sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "skills": "",
        }

        # Simple section detection (in production, use better parsing)
        lines = text.split("\n")

        current_section = None
        for line in lines:
            line_lower = line.lower().strip()

            if any(kw in line_lower for kw in ["summary", "objective", "profile"]):
                current_section = "summary"
            elif any(kw in line_lower for kw in ["experience", "employment", "work"]):
                current_section = "experience"
            elif any(kw in line_lower for kw in ["education", "academic"]):
                current_section = "education"
            elif any(kw in line_lower for kw in ["skills", "competencies", "technologies"]):
                current_section = "skills"

            if current_section:
                sections[current_section] += line + "\n"

        return sections

    def extract_bert_features(self, parsed_data, job_requirements: Optional[str] = None) -> Dict[str, float]:
        """
        Extract BERT-based features from parsed resume.

        Args:
            parsed_data: ParsedData object
            job_requirements: Optional job description/requirements for semantic matching

        Returns:
            Dictionary of BERT features
        """
        if not self.model:
            return {}

        features = {}

        # Summary embedding features
        if parsed_data.summary:
            summary_emb = self.extract_embedding(parsed_data.summary)
            if summary_emb:
                import numpy as np
                features["bert_summary_norm"] = float(np.linalg.norm(summary_emb))
                features["bert_summary_mean"] = float(np.mean(summary_emb))
                features["bert_summary_std"] = float(np.std(summary_emb))

        # Experience embeddings
        if parsed_data.experiences:
            exp_texts = [
                f"{exp.title} at {exp.company}. {exp.description or ''}"
                for exp in parsed_data.experiences[:3]  # Limit to first 3
            ]

            exp_embeddings = [self.extract_embedding(text) for text in exp_texts]
            exp_embeddings = [e for e in exp_embeddings if e is not None]

            if exp_embeddings:
                import numpy as np
                # Average embedding
                avg_emb = np.mean(exp_embeddings, axis=0)
                features["bert_exp_avg_norm"] = float(np.linalg.norm(avg_emb))

                # Diversity of experiences (variance)
                if len(exp_embeddings) > 1:
                    features["bert_exp_diversity"] = float(np.var([np.linalg.norm(e) for e in exp_embeddings]))

        # Skills embedding
        if parsed_data.technical_skills:
            skills_text = ", ".join(parsed_data.technical_skills[:20])  # Limit length
            skills_emb = self.extract_embedding(skills_text)
            if skills_emb:
                import numpy as np
                features["bert_skills_norm"] = float(np.linalg.norm(skills_emb))

        # Skills Semantic Match Features (0.67 validity - HIGHEST PREDICTOR)
        if job_requirements:
            semantic_features = self.extract_skills_semantic_match_features(
                parsed_data, job_requirements
            )
            features.update(semantic_features)

        logger.info(f"Extracted {len(features)} BERT features")

        return features

    def extract_skills_semantic_match_features(
        self,
        parsed_data,
        job_requirements: str
    ) -> Dict[str, float]:
        """
        Extract Skills Semantic Match features.

        Validity: 0.67 (HIGHEST PREDICTOR)
        Research: Alonso et al. (2025) - Semantic matching outperformed keyword at 92% accuracy
                  BERT/LLaMA models achieve 89-95% accuracy in resume-job matching

        This is THE most important predictor based on meta-analytic research.

        Args:
            parsed_data: ParsedData object
            job_requirements: Job description/requirements text

        Returns:
            Dictionary of semantic match features
        """
        if not self.model:
            return {
                "semantic_skills_match_overall": 0.5,
                "semantic_skills_cosine_similarity": 0.5,
                "semantic_experience_match": 0.5,
                "semantic_technical_depth_match": 0.5,
                "semantic_role_fit": 0.5,
            }

        features = {}

        try:
            import numpy as np

            # 1. Overall Skills Semantic Match (Cosine Similarity)
            # Combine all candidate skills
            all_skills = list(parsed_data.technical_skills) + [s.name for s in parsed_data.skills]
            candidate_skills_text = ", ".join(all_skills[:30])  # Limit to 30 skills

            if candidate_skills_text and job_requirements:
                # Get embeddings
                candidate_emb = self.extract_embedding(candidate_skills_text)
                job_emb = self.extract_embedding(job_requirements)

                if candidate_emb is not None and job_emb is not None:
                    candidate_emb = np.array(candidate_emb)
                    job_emb = np.array(job_emb)

                    # Cosine similarity (0-1, research shows 0.74-0.83 for good matches)
                    cosine_sim = np.dot(candidate_emb, job_emb) / (
                        np.linalg.norm(candidate_emb) * np.linalg.norm(job_emb)
                    )
                    features["semantic_skills_cosine_similarity"] = float(cosine_sim)

                    # Overall match score (0-100)
                    # Research: semantic match at 0.67 validity vs keyword at 0.35
                    features["semantic_skills_match_overall"] = float(cosine_sim * 100)
                else:
                    features["semantic_skills_cosine_similarity"] = 0.5
                    features["semantic_skills_match_overall"] = 50.0
            else:
                features["semantic_skills_cosine_similarity"] = 0.5
                features["semantic_skills_match_overall"] = 50.0

            # 2. Experience-Job Semantic Match
            # Compare experience descriptions to job requirements
            if parsed_data.experiences:
                exp_texts = [
                    f"{exp.title}. {exp.description or ''}"
                    for exp in parsed_data.experiences[:3]
                ]
                combined_exp = " ".join(exp_texts)

                if combined_exp and job_requirements:
                    exp_job_similarity = self.calculate_semantic_similarity(
                        combined_exp, job_requirements
                    )
                    features["semantic_experience_match"] = float(exp_job_similarity * 100)
                else:
                    features["semantic_experience_match"] = 50.0
            else:
                features["semantic_experience_match"] = 50.0

            # 3. Technical Depth Match
            # Match depth of technical expertise to job requirements
            technical_text = ", ".join(parsed_data.technical_skills[:20])
            if technical_text and job_requirements:
                tech_similarity = self.calculate_semantic_similarity(
                    technical_text, job_requirements
                )
                features["semantic_technical_depth_match"] = float(tech_similarity * 100)
            else:
                features["semantic_technical_depth_match"] = 50.0

            # 4. Role Fit (Title/Role semantic match)
            if parsed_data.experiences:
                recent_title = parsed_data.experiences[0].title if parsed_data.experiences[0].title else ""
                if recent_title and job_requirements:
                    role_similarity = self.calculate_semantic_similarity(
                        recent_title, job_requirements
                    )
                    features["semantic_role_fit"] = float(role_similarity * 100)
                else:
                    features["semantic_role_fit"] = 50.0
            else:
                features["semantic_role_fit"] = 50.0

            # 5. Semantic Match Quality Score (composite)
            # Weight: skills (40%) + experience (30%) + technical depth (20%) + role (10%)
            match_quality = (
                features["semantic_skills_match_overall"] * 0.40 +
                features["semantic_experience_match"] * 0.30 +
                features["semantic_technical_depth_match"] * 0.20 +
                features["semantic_role_fit"] * 0.10
            )
            features["semantic_match_quality_score"] = match_quality

            # 6. Match Confidence (based on research accuracy)
            # Research shows 89-95% accuracy for BERT/LLaMA semantic matching
            # Confidence is higher when cosine similarity is in expected range (0.74-0.83 for good matches)
            cosine = features["semantic_skills_cosine_similarity"]
            if 0.74 <= cosine <= 0.95:
                confidence = 0.95  # High confidence - in research-validated range
            elif 0.60 <= cosine < 0.74:
                confidence = 0.85  # Good confidence
            elif 0.40 <= cosine < 0.60:
                confidence = 0.70  # Moderate confidence
            else:
                confidence = 0.60  # Lower confidence

            features["semantic_match_confidence"] = confidence * 100

            logger.info(f"Extracted {len(features)} semantic match features")
            logger.info(f"  Overall semantic match: {features['semantic_skills_match_overall']:.1f}%")
            logger.info(f"  Cosine similarity: {features['semantic_skills_cosine_similarity']:.3f}")

        except Exception as e:
            logger.error(f"Error extracting semantic match features: {e}")
            # Return default values on error
            features = {
                "semantic_skills_match_overall": 50.0,
                "semantic_skills_cosine_similarity": 0.5,
                "semantic_experience_match": 50.0,
                "semantic_technical_depth_match": 50.0,
                "semantic_role_fit": 50.0,
                "semantic_match_quality_score": 50.0,
                "semantic_match_confidence": 60.0,
            }

        return features


class TransformerResumeClassifier:
    """
    Classify resumes using fine-tuned transformer models.

    Can be trained to:
    - Predict seniority level
    - Classify role type
    - Estimate salary range
    - Predict job fit
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize classifier.

        Args:
            model_path: Path to fine-tuned model (uses base if None)
        """
        self.model_path = model_path or "bert-base-uncased"
        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load model for classification."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

            logger.info(f"✅ Loaded classifier: {self.model_path}")

        except Exception as e:
            logger.warning(f"⚠️  Could not load classifier: {e}")

    def predict_seniority(self, resume_text: str) -> Dict[str, float]:
        """
        Predict seniority level from resume.

        Returns:
            Dictionary of {level: probability}
        """
        if not self.model:
            # Fallback to rule-based
            return self._predict_seniority_rules(resume_text)

        try:
            import torch
            from torch.nn.functional import softmax

            # Tokenize
            inputs = self.tokenizer(
                resume_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=-1)[0]

            # Map to levels
            levels = ["entry", "junior", "mid", "senior", "lead", "executive"]
            predictions = {
                level: float(prob)
                for level, prob in zip(levels, probs)
            }

            return predictions

        except Exception as e:
            logger.error(f"Error predicting seniority: {e}")
            return self._predict_seniority_rules(resume_text)

    def _predict_seniority_rules(self, resume_text: str) -> Dict[str, float]:
        """Fallback rule-based seniority prediction."""
        text_lower = resume_text.lower()

        scores = {
            "entry": 0.0,
            "junior": 0.0,
            "mid": 0.0,
            "senior": 0.0,
            "lead": 0.0,
            "executive": 0.0,
        }

        # Count keywords
        if "senior" in text_lower:
            scores["senior"] += 0.4
        if "lead" in text_lower or "principal" in text_lower:
            scores["lead"] += 0.4
        if any(title in text_lower for title in ["director", "vp", "chief", "cto", "ceo"]):
            scores["executive"] += 0.5
        if "junior" in text_lower or "associate" in text_lower:
            scores["junior"] += 0.3

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        else:
            # Default to mid
            scores["mid"] = 1.0

        return scores


# Utility functions
def create_bert_feature_extractor(use_gpu: bool = False) -> Optional[BERTFeatureExtractor]:
    """
    Create BERT feature extractor with error handling.

    Args:
        use_gpu: Use GPU if available

    Returns:
        BERTFeatureExtractor or None if not available
    """
    try:
        extractor = BERTFeatureExtractor(use_gpu=use_gpu)
        if extractor.model:
            return extractor
        return None
    except Exception as e:
        logger.warning(f"Could not create BERT extractor: {e}")
        return None


def calculate_resume_job_similarity_bert(
    resume_text: str,
    job_description: str,
    extractor: Optional[BERTFeatureExtractor] = None
) -> float:
    """
    Calculate similarity between resume and job description using BERT.

    Args:
        resume_text: Resume text
        job_description: Job description text
        extractor: BERT extractor (creates new if None)

    Returns:
        Similarity score (0-1)
    """
    if extractor is None:
        extractor = create_bert_feature_extractor()

    if extractor is None:
        return 0.5  # Default

    return extractor.calculate_semantic_similarity(resume_text, job_description)
