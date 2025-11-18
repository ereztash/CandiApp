"""
Feature Store for Resume Features

Provides persistent storage and efficient retrieval of engineered features.
Supports multiple backends (JSON, pickle, database) and caching.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from .features import FeatureVector

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature store for persisting and retrieving engineered features.

    Supports multiple storage backends and provides caching for performance.
    """

    def __init__(self, storage_path: str = "./feature_store", backend: str = "json"):
        """
        Initialize feature store.

        Args:
            storage_path: Path to store features
            backend: Storage backend ("json", "pickle")
        """
        self.storage_path = Path(storage_path)
        self.backend = backend
        self.cache: Dict[str, FeatureVector] = {}

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized feature store at {self.storage_path} with {backend} backend")

    def save(self, resume_id: str, feature_vector: FeatureVector):
        """
        Save a feature vector to the store.

        Args:
            resume_id: Unique identifier for the resume
            feature_vector: FeatureVector to save
        """
        feature_path = self._get_feature_path(resume_id)

        if self.backend == "json":
            # Save as JSON
            data = feature_vector.to_dict()
            with open(feature_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif self.backend == "pickle":
            # Save as pickle
            with open(feature_path, 'wb') as f:
                pickle.dump(feature_vector, f)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Update cache
        self.cache[resume_id] = feature_vector

        logger.debug(f"Saved feature vector for resume {resume_id}")

    def load(self, resume_id: str) -> Optional[FeatureVector]:
        """
        Load a feature vector from the store.

        Args:
            resume_id: Unique identifier for the resume

        Returns:
            FeatureVector if found, None otherwise
        """
        # Check cache first
        if resume_id in self.cache:
            logger.debug(f"Retrieved feature vector for resume {resume_id} from cache")
            return self.cache[resume_id]

        feature_path = self._get_feature_path(resume_id)

        if not feature_path.exists():
            logger.warning(f"Feature vector not found for resume {resume_id}")
            return None

        if self.backend == "json":
            # Load from JSON
            with open(feature_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct FeatureVector
            feature_vector = FeatureVector(
                numerical_features=data.get("numerical", {}),
                categorical_features=data.get("categorical", {}),
                text_features=data.get("text", {}),
                list_features=data.get("list", {}),
                feature_count=data.get("metadata", {}).get("feature_count", 0),
            )

            # Parse timestamp if present
            timestamp_str = data.get("metadata", {}).get("extraction_timestamp")
            if timestamp_str:
                feature_vector.extraction_timestamp = datetime.fromisoformat(timestamp_str)

        elif self.backend == "pickle":
            # Load from pickle
            with open(feature_path, 'rb') as f:
                feature_vector = pickle.load(f)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Update cache
        self.cache[resume_id] = feature_vector

        logger.debug(f"Loaded feature vector for resume {resume_id}")

        return feature_vector

    def delete(self, resume_id: str):
        """
        Delete a feature vector from the store.

        Args:
            resume_id: Unique identifier for the resume
        """
        feature_path = self._get_feature_path(resume_id)

        if feature_path.exists():
            feature_path.unlink()
            logger.debug(f"Deleted feature vector for resume {resume_id}")

        # Remove from cache
        if resume_id in self.cache:
            del self.cache[resume_id]

    def list_all(self) -> List[str]:
        """
        List all resume IDs in the store.

        Returns:
            List of resume IDs
        """
        if self.backend == "json":
            pattern = "*.json"
        elif self.backend == "pickle":
            pattern = "*.pkl"
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        resume_ids = []
        for path in self.storage_path.glob(pattern):
            resume_id = path.stem
            resume_ids.append(resume_id)

        return resume_ids

    def batch_save(self, features: Dict[str, FeatureVector]):
        """
        Save multiple feature vectors in batch.

        Args:
            features: Dictionary mapping resume_id to FeatureVector
        """
        logger.info(f"Batch saving {len(features)} feature vectors...")

        for resume_id, feature_vector in features.items():
            self.save(resume_id, feature_vector)

        logger.info("Batch save completed")

    def batch_load(self, resume_ids: List[str]) -> Dict[str, FeatureVector]:
        """
        Load multiple feature vectors in batch.

        Args:
            resume_ids: List of resume IDs to load

        Returns:
            Dictionary mapping resume_id to FeatureVector
        """
        logger.info(f"Batch loading {len(resume_ids)} feature vectors...")

        features = {}
        for resume_id in resume_ids:
            feature_vector = self.load(resume_id)
            if feature_vector:
                features[resume_id] = feature_vector

        logger.info(f"Batch load completed ({len(features)}/{len(resume_ids)} found)")

        return features

    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()
        logger.debug("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feature store.

        Returns:
            Dictionary with store statistics
        """
        all_ids = self.list_all()

        stats = {
            "total_features": len(all_ids),
            "cache_size": len(self.cache),
            "storage_path": str(self.storage_path),
            "backend": self.backend,
        }

        # Calculate total storage size
        total_size = 0
        for resume_id in all_ids:
            path = self._get_feature_path(resume_id)
            if path.exists():
                total_size += path.stat().st_size

        stats["total_storage_bytes"] = total_size
        stats["total_storage_mb"] = total_size / (1024 * 1024)

        return stats

    def _get_feature_path(self, resume_id: str) -> Path:
        """
        Get the file path for a resume's features.

        Args:
            resume_id: Unique identifier for the resume

        Returns:
            Path to feature file
        """
        if self.backend == "json":
            filename = f"{resume_id}.json"
        elif self.backend == "pickle":
            filename = f"{resume_id}.pkl"
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        return self.storage_path / filename


class FeatureIndex:
    """
    Index for fast feature lookup and search.

    Maintains an in-memory index of feature metadata for quick searches.
    """

    def __init__(self):
        """Initialize feature index."""
        self.index: Dict[str, Dict[str, Any]] = {}

    def add(self, resume_id: str, feature_vector: FeatureVector):
        """
        Add a feature vector to the index.

        Args:
            resume_id: Unique identifier for the resume
            feature_vector: FeatureVector to index
        """
        # Store metadata for fast lookup
        self.index[resume_id] = {
            "feature_count": feature_vector.feature_count,
            "extraction_timestamp": feature_vector.extraction_timestamp,
            "numerical_feature_names": list(feature_vector.numerical_features.keys()),
            "categorical_feature_names": list(feature_vector.categorical_features.keys()),
        }

    def search_by_features(self, required_features: List[str]) -> List[str]:
        """
        Search for resumes that have all required features.

        Args:
            required_features: List of feature names that must be present

        Returns:
            List of resume IDs that have all required features
        """
        matching_ids = []

        for resume_id, metadata in self.index.items():
            all_features = set(metadata["numerical_feature_names"] + metadata["categorical_feature_names"])

            if all(feat in all_features for feat in required_features):
                matching_ids.append(resume_id)

        return matching_ids

    def get_all_feature_names(self) -> List[str]:
        """
        Get all unique feature names across all indexed resumes.

        Returns:
            List of unique feature names
        """
        all_features = set()

        for metadata in self.index.values():
            all_features.update(metadata["numerical_feature_names"])
            all_features.update(metadata["categorical_feature_names"])

        return sorted(all_features)

    def get_feature_coverage(self) -> Dict[str, int]:
        """
        Get coverage statistics for each feature.

        Returns:
            Dictionary mapping feature name to count of resumes with that feature
        """
        coverage = {}

        for metadata in self.index.values():
            for feature_name in metadata["numerical_feature_names"] + metadata["categorical_feature_names"]:
                coverage[feature_name] = coverage.get(feature_name, 0) + 1

        return coverage
