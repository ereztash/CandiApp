#!/usr/bin/env python3
"""
CandiApp v2.0 GPU Benchmark & Comparison
=========================================

Compares performance between:
1. v1.0: Base features (127) - CPU only
2. v2.0: Advanced features (219) - CPU + GPU acceleration

Features tested:
- Base feature extraction (127 features)
- Advanced features (92 additional features)
- BERT-based features with GPU
- Semantic skill matching
- Advanced NLP with spaCy large models
- Multilingual processing

Usage:
    python run_v2_comparison.py --samples 500 --gpu
    python run_v2_comparison.py --quick
"""

import argparse
import sys
import json
import logging
import time
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

# PyTorch for GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available. GPU benchmarks will be skipped.")

from candiapp.features import FeatureExtractor, FeatureTransformer
from candiapp.models import ParsedData

# Advanced features (v2.0)
try:
    from candiapp.advanced_features import AdvancedFeatureExtractor
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False
    print("⚠️  Advanced features not available")

try:
    from candiapp.bert_features import BERTFeatureExtractor
    HAS_BERT = True
except ImportError:
    HAS_BERT = False
    print("⚠️  BERT features not available")

try:
    from candiapp.semantic_matching import SemanticSkillMatcher
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    print("⚠️  Semantic matching not available")

try:
    from candiapp.nlp_advanced import AdvancedNLPProcessor
    HAS_NLP_ADVANCED = True
except ImportError:
    HAS_NLP_ADVANCED = False
    print("⚠️  Advanced NLP not available")

from synthetic_data_generator import SyntheticResumeGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class V2BenchmarkSuite:
    """Benchmark suite for v2.0 comparison."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.results = {
            "v1_base": [],
            "v2_advanced": [],
            "gpu_comparison": [],
            "feature_comparison": [],
            "module_breakdown": []
        }

        # Initialize extractors
        self.base_extractor = FeatureExtractor()

        if HAS_ADVANCED:
            self.advanced_extractor = AdvancedFeatureExtractor()

        if HAS_BERT and self.use_gpu:
            self.bert_extractor = BERTFeatureExtractor(use_gpu=True)
        elif HAS_BERT:
            self.bert_extractor = BERTFeatureExtractor(use_gpu=False)

        if HAS_SEMANTIC:
            self.semantic_matcher = SemanticSkillMatcher(use_embeddings=False)

        if HAS_NLP_ADVANCED:
            try:
                self.nlp_processor = AdvancedNLPProcessor()
            except:
                self.nlp_processor = None

    def benchmark_base_features(self, samples: List[ParsedData]) -> Dict[str, Any]:
        """Benchmark v1.0 base features (127 features)."""
        logger.info("\n" + "="*80)
        logger.info("📊 Benchmarking v1.0 BASE FEATURES (127 features)")
        logger.info("="*80)

        results = {}

        # Warmup
        for i in range(min(10, len(samples))):
            _ = self.base_extractor.extract_features(samples[i])

        # Timing test
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        feature_vectors = []
        for sample in samples:
            fv = self.base_extractor.extract_features(sample)
            feature_vectors.append(fv)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        elapsed = end_time - start_time
        throughput = len(samples) / elapsed
        memory_used = end_memory - start_memory

        # Count features
        feature_count = len(feature_vectors[0].numerical_features) if feature_vectors else 0

        results = {
            "version": "v1.0 - Base Features",
            "feature_count": feature_count,
            "samples_processed": len(samples),
            "total_time_sec": elapsed,
            "throughput_samples_per_sec": throughput,
            "avg_time_per_sample_ms": (elapsed / len(samples)) * 1000,
            "memory_used_mb": memory_used,
            "device": "CPU"
        }

        logger.info(f"✅ Processed {len(samples)} samples in {elapsed:.2f}s")
        logger.info(f"   • Throughput: {throughput:.2f} samples/sec")
        logger.info(f"   • Avg time: {results['avg_time_per_sample_ms']:.2f} ms/sample")
        logger.info(f"   • Feature count: {feature_count}")
        logger.info(f"   • Memory: {memory_used:.2f} MB")

        return results

    def benchmark_advanced_features(self, samples: List[ParsedData],
                                   use_bert: bool = False,
                                   use_nlp_advanced: bool = False) -> Dict[str, Any]:
        """Benchmark v2.0 advanced features (219 features)."""
        logger.info("\n" + "="*80)
        config = []
        if use_bert:
            config.append("BERT")
        if use_nlp_advanced:
            config.append("Advanced NLP")
        config_str = " + ".join(config) if config else "Standard"
        logger.info(f"📊 Benchmarking v2.0 ADVANCED FEATURES ({config_str})")
        logger.info("="*80)

        if not HAS_ADVANCED:
            logger.warning("⚠️  Advanced features not available, skipping")
            return {}

        results = {}

        # Warmup
        for i in range(min(5, len(samples))):
            base_fv = self.base_extractor.extract_features(samples[i])
            base_features_dict = {**base_fv.numerical_features,
                                 **base_fv.categorical_features}
            _ = self.advanced_extractor.extract_advanced_features(samples[i], base_features_dict)

        # Timing test
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        if self.use_gpu and use_bert:
            torch.cuda.synchronize()

        feature_vectors = []
        bert_features_list = []
        nlp_entities_list = []

        for sample in samples:
            # Base features
            base_fv = self.base_extractor.extract_features(sample)
            base_features_dict = {**base_fv.numerical_features,
                                 **base_fv.categorical_features}

            # Advanced features
            advanced_features = self.advanced_extractor.extract_advanced_features(
                sample, base_features_dict
            )

            # BERT features (if enabled)
            if use_bert and HAS_BERT and hasattr(self, 'bert_extractor'):
                try:
                    # Get resume summary text
                    summary = getattr(sample, 'summary', '') or ''
                    if summary:
                        bert_feats = self.bert_extractor.extract_features(summary)
                        bert_features_list.append(bert_feats)
                except Exception as e:
                    pass

            # Advanced NLP (if enabled)
            if use_nlp_advanced and HAS_NLP_ADVANCED and self.nlp_processor:
                try:
                    text = getattr(sample, 'summary', '') or ''
                    if text:
                        entities = self.nlp_processor.extract_entities(text)
                        nlp_entities_list.append(entities)
                except Exception as e:
                    pass

            feature_vectors.append(advanced_features)

        if self.use_gpu and use_bert:
            torch.cuda.synchronize()

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        elapsed = end_time - start_time
        throughput = len(samples) / elapsed
        memory_used = end_memory - start_memory

        # Count features
        total_features = 127  # base
        if feature_vectors:
            total_features += len(feature_vectors[0])
        if use_bert and bert_features_list:
            total_features += len(bert_features_list[0].get('numerical_features', {}))

        results = {
            "version": "v2.0 - Advanced Features",
            "feature_count": total_features,
            "samples_processed": len(samples),
            "total_time_sec": elapsed,
            "throughput_samples_per_sec": throughput,
            "avg_time_per_sample_ms": (elapsed / len(samples)) * 1000,
            "memory_used_mb": memory_used,
            "device": "GPU" if (self.use_gpu and use_bert) else "CPU",
            "bert_enabled": use_bert and HAS_BERT,
            "advanced_nlp_enabled": use_nlp_advanced and HAS_NLP_ADVANCED
        }

        logger.info(f"✅ Processed {len(samples)} samples in {elapsed:.2f}s")
        logger.info(f"   • Throughput: {throughput:.2f} samples/sec")
        logger.info(f"   • Avg time: {results['avg_time_per_sample_ms']:.2f} ms/sample")
        logger.info(f"   • Feature count: {total_features}")
        logger.info(f"   • Memory: {memory_used:.2f} MB")
        logger.info(f"   • Device: {results['device']}")

        return results

    def benchmark_module_breakdown(self, samples: List[ParsedData]) -> Dict[str, Any]:
        """Benchmark individual module performance."""
        logger.info("\n" + "="*80)
        logger.info("🔬 Module Performance Breakdown")
        logger.info("="*80)

        breakdown = {}

        # Take subset for detailed analysis
        test_samples = samples[:min(100, len(samples))]

        # 1. Base features
        start = time.time()
        for sample in test_samples:
            _ = self.base_extractor.extract_features(sample)
        base_time = (time.time() - start) / len(test_samples) * 1000
        breakdown["base_features"] = {"avg_ms": base_time, "feature_count": 127}
        logger.info(f"   • Base features: {base_time:.2f} ms/sample (127 features)")

        # 2. Advanced features
        if HAS_ADVANCED:
            start = time.time()
            for sample in test_samples:
                base_fv = self.base_extractor.extract_features(sample)
                base_dict = {**base_fv.numerical_features, **base_fv.categorical_features}
                _ = self.advanced_extractor.extract_advanced_features(sample, base_dict)
            advanced_time = (time.time() - start) / len(test_samples) * 1000
            breakdown["advanced_features"] = {"avg_ms": advanced_time, "feature_count": 92}
            logger.info(f"   • Advanced features: {advanced_time:.2f} ms/sample (92 features)")

        # 3. BERT features (CPU)
        if HAS_BERT:
            bert_cpu = BERTFeatureExtractor(use_gpu=False)
            start = time.time()
            for sample in test_samples:
                text = getattr(sample, 'summary', '') or 'Software engineer with Python'
                try:
                    _ = bert_cpu.extract_features(text)
                except:
                    pass
            bert_cpu_time = (time.time() - start) / len(test_samples) * 1000
            breakdown["bert_cpu"] = {"avg_ms": bert_cpu_time, "device": "CPU"}
            logger.info(f"   • BERT (CPU): {bert_cpu_time:.2f} ms/sample")

            # BERT (GPU)
            if self.use_gpu:
                bert_gpu = BERTFeatureExtractor(use_gpu=True)
                torch.cuda.synchronize()
                start = time.time()
                for sample in test_samples:
                    text = getattr(sample, 'summary', '') or 'Software engineer with Python'
                    try:
                        _ = bert_gpu.extract_features(text)
                    except:
                        pass
                torch.cuda.synchronize()
                bert_gpu_time = (time.time() - start) / len(test_samples) * 1000
                breakdown["bert_gpu"] = {"avg_ms": bert_gpu_time, "device": "GPU"}
                speedup = bert_cpu_time / bert_gpu_time
                logger.info(f"   • BERT (GPU): {bert_gpu_time:.2f} ms/sample ({speedup:.1f}x speedup)")

        # 4. Semantic matching
        if HAS_SEMANTIC:
            start = time.time()
            test_skills = ["python", "javascript", "react", "machine learning", "docker"]
            for _ in range(len(test_samples)):
                for i, skill1 in enumerate(test_skills):
                    for skill2 in test_skills[i+1:]:
                        _ = self.semantic_matcher.calculate_skill_similarity(skill1, skill2)
            semantic_time = (time.time() - start) / len(test_samples) * 1000
            breakdown["semantic_matching"] = {"avg_ms": semantic_time}
            logger.info(f"   • Semantic matching: {semantic_time:.2f} ms/sample")

        # 5. Advanced NLP
        if HAS_NLP_ADVANCED and self.nlp_processor:
            start = time.time()
            for sample in test_samples:
                text = getattr(sample, 'summary', '') or 'Worked at Google from 2020 to 2023'
                try:
                    _ = self.nlp_processor.extract_entities(text)
                    _ = self.nlp_processor.extract_dates(text)
                except:
                    pass
            nlp_time = (time.time() - start) / len(test_samples) * 1000
            breakdown["advanced_nlp"] = {"avg_ms": nlp_time}
            logger.info(f"   • Advanced NLP: {nlp_time:.2f} ms/sample")

        return breakdown

    def run_full_comparison(self, samples: List[ParsedData]) -> Dict[str, Any]:
        """Run complete v1 vs v2 comparison."""
        logger.info("\n" + "="*80)
        logger.info("🎯 FULL COMPARISON: v1.0 vs v2.0")
        logger.info("="*80)

        all_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "samples": len(samples),
                "gpu_available": self.use_gpu,
                "gpu_name": torch.cuda.get_device_name(0) if self.use_gpu else None,
                "modules_available": {
                    "advanced_features": HAS_ADVANCED,
                    "bert": HAS_BERT,
                    "semantic_matching": HAS_SEMANTIC,
                    "advanced_nlp": HAS_NLP_ADVANCED
                }
            }
        }

        # 1. Base features (v1.0)
        v1_results = self.benchmark_base_features(samples)
        all_results["v1_base"] = v1_results

        # 2. Advanced features CPU (v2.0)
        v2_cpu_results = self.benchmark_advanced_features(
            samples, use_bert=False, use_nlp_advanced=True
        )
        all_results["v2_cpu"] = v2_cpu_results

        # 3. Advanced features GPU (v2.0)
        if self.use_gpu and HAS_BERT:
            v2_gpu_results = self.benchmark_advanced_features(
                samples, use_bert=True, use_nlp_advanced=True
            )
            all_results["v2_gpu"] = v2_gpu_results

        # 4. Module breakdown
        breakdown = self.benchmark_module_breakdown(samples)
        all_results["module_breakdown"] = breakdown

        # 5. Comparison summary
        logger.info("\n" + "="*80)
        logger.info("📊 COMPARISON SUMMARY")
        logger.info("="*80)

        if v1_results and v2_cpu_results:
            logger.info("\n🔹 v1.0 Base vs v2.0 Advanced (CPU):")

            feature_increase = v2_cpu_results["feature_count"] - v1_results["feature_count"]
            feature_increase_pct = (feature_increase / v1_results["feature_count"]) * 100
            logger.info(f"   • Features: {v1_results['feature_count']} → {v2_cpu_results['feature_count']} (+{feature_increase}, +{feature_increase_pct:.1f}%)")

            throughput_change = ((v2_cpu_results["throughput_samples_per_sec"] -
                                v1_results["throughput_samples_per_sec"]) /
                               v1_results["throughput_samples_per_sec"]) * 100
            logger.info(f"   • Throughput: {v1_results['throughput_samples_per_sec']:.2f} → {v2_cpu_results['throughput_samples_per_sec']:.2f} samples/sec ({throughput_change:+.1f}%)")

            time_change = ((v2_cpu_results["avg_time_per_sample_ms"] -
                          v1_results["avg_time_per_sample_ms"]) /
                         v1_results["avg_time_per_sample_ms"]) * 100
            logger.info(f"   • Avg time: {v1_results['avg_time_per_sample_ms']:.2f} → {v2_cpu_results['avg_time_per_sample_ms']:.2f} ms/sample ({time_change:+.1f}%)")

            all_results["comparison_v1_v2_cpu"] = {
                "feature_increase": feature_increase,
                "feature_increase_pct": feature_increase_pct,
                "throughput_change_pct": throughput_change,
                "time_change_pct": time_change
            }

        if self.use_gpu and "v2_gpu" in all_results:
            logger.info("\n🔹 v2.0 CPU vs GPU:")
            v2_gpu = all_results["v2_gpu"]

            gpu_speedup = v2_cpu_results["avg_time_per_sample_ms"] / v2_gpu["avg_time_per_sample_ms"]
            logger.info(f"   • Avg time: {v2_cpu_results['avg_time_per_sample_ms']:.2f} → {v2_gpu['avg_time_per_sample_ms']:.2f} ms/sample ({gpu_speedup:.2f}x speedup)")

            throughput_increase = ((v2_gpu["throughput_samples_per_sec"] -
                                  v2_cpu_results["throughput_samples_per_sec"]) /
                                 v2_cpu_results["throughput_samples_per_sec"]) * 100
            logger.info(f"   • Throughput: {v2_cpu_results['throughput_samples_per_sec']:.2f} → {v2_gpu['throughput_samples_per_sec']:.2f} samples/sec (+{throughput_increase:.1f}%)")

            all_results["comparison_cpu_gpu"] = {
                "gpu_speedup": gpu_speedup,
                "throughput_increase_pct": throughput_increase
            }

        return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CandiApp v2.0 GPU Benchmark & Comparison"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of synthetic resumes to generate (default: 500)"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration if available (default: True)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_false",
        dest="gpu",
        help="Disable GPU acceleration"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (100 samples)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )

    return parser.parse_args()


def main():
    """Main benchmark execution."""
    args = parse_args()

    # Adjust samples for quick mode
    if args.quick:
        n_samples = 100
        logger.info("🏃 Running in QUICK mode")
    else:
        n_samples = args.samples

    logger.info("="*80)
    logger.info("🎯 CANDIAPP v2.0 GPU BENCHMARK & COMPARISON")
    logger.info("="*80)
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Samples: {n_samples}")
    logger.info(f"   - GPU: {'Enabled' if args.gpu else 'Disabled'}")
    logger.info("="*80)

    # Check GPU availability
    if args.gpu and TORCH_AVAILABLE:
        if torch.cuda.is_available():
            logger.info(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            logger.info(f"   - CUDA Version: {torch.version.cuda}")
            logger.info(f"   - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("⚠️  GPU requested but not available, falling back to CPU")
            args.gpu = False
    else:
        logger.info("💻 Running on CPU")

    # Generate synthetic dataset
    logger.info("\n" + "="*80)
    logger.info("📝 Generating Synthetic Dataset")
    logger.info("="*80)

    generator = SyntheticResumeGenerator()
    resumes, _ = generator.generate_dataset(n_samples=n_samples)

    logger.info(f"✅ Generated {len(resumes)} synthetic resumes")

    # Run benchmarks
    suite = V2BenchmarkSuite(use_gpu=args.gpu)
    results = suite.run_full_comparison(resumes)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"v2_comparison_results_{timestamp}.json"

    logger.info(f"\n💾 Saving results to {output_file}")

    # Convert to JSON-serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=convert_to_serializable)

    logger.info("✅ Results saved successfully")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("✨ BENCHMARK COMPLETED")
    logger.info("="*80)

    if "v1_base" in results and "v2_cpu" in results:
        v1 = results["v1_base"]
        v2 = results["v2_cpu"]

        logger.info("\n🏆 Key Improvements in v2.0:")
        logger.info(f"   ✓ Feature count increased: {v1['feature_count']} → {v2['feature_count']} features")
        logger.info(f"   ✓ Processing time: {v1['avg_time_per_sample_ms']:.2f} → {v2['avg_time_per_sample_ms']:.2f} ms/sample")

        if "v2_gpu" in results:
            v2_gpu = results["v2_gpu"]
            logger.info(f"   ✓ With GPU acceleration: {v2_gpu['avg_time_per_sample_ms']:.2f} ms/sample")
            speedup = v1['avg_time_per_sample_ms'] / v2_gpu['avg_time_per_sample_ms']
            logger.info(f"   ✓ Overall speedup: {speedup:.2f}x faster")

    logger.info("\n" + "="*80)
    logger.info(f"📁 Full results saved to: {output_file}")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
