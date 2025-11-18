#!/usr/bin/env python3
"""
Quick Benchmark Demo

Runs a lightweight version of benchmarks without heavy dependencies.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synthetic_data_generator import SyntheticResumeGenerator
from candiapp.features import FeatureExtractor, FeatureTransformer, create_feature_pipeline
from candiapp.models import ExperienceLevel
import numpy as np


def run_quick_benchmark():
    """Run quick benchmark without PyTorch."""
    print("="*80)
    print("🎯 CANDIAPP FEATURE ENGINEERING - QUICK BENCHMARK")
    print("="*80)

    # Generate synthetic data
    print("\n📝 Step 1: Generating synthetic resumes...")
    generator = SyntheticResumeGenerator()

    distribution = {
        ExperienceLevel.ENTRY: 0.10,
        ExperienceLevel.JUNIOR: 0.15,
        ExperienceLevel.MID: 0.30,
        ExperienceLevel.SENIOR: 0.25,
        ExperienceLevel.LEAD: 0.15,
        ExperienceLevel.EXECUTIVE: 0.05,
    }

    n_samples = 200
    resumes, labels = generator.generate_dataset(
        n_samples=n_samples,
        distribution=distribution
    )

    print(f"✅ Generated {len(resumes)} synthetic resumes")

    # Benchmark 1: Feature Extraction Speed
    print("\n⚡ Benchmark 1: Feature Extraction Speed")
    extractor = FeatureExtractor()

    start_time = time.time()
    feature_vectors = []
    for resume in resumes:
        fv = extractor.extract_features(resume)
        feature_vectors.append(fv)

    elapsed = time.time() - start_time
    throughput = len(resumes) / elapsed

    print(f"   ✓ Time: {elapsed:.2f}s")
    print(f"   ✓ Throughput: {throughput:.2f} resumes/sec")
    print(f"   ✓ Avg time per resume: {elapsed/len(resumes)*1000:.2f} ms")

    # Benchmark 2: Feature Coverage
    print("\n📊 Benchmark 2: Feature Coverage")
    feature_counts = [fv.feature_count for fv in feature_vectors]
    avg_features = np.mean(feature_counts)
    max_features = np.max(feature_counts)
    min_features = np.min(feature_counts)

    print(f"   ✓ Average features: {avg_features:.1f}")
    print(f"   ✓ Max features: {max_features}")
    print(f"   ✓ Min features: {min_features}")

    # Benchmark 3: Feature Transformation Speed
    print("\n🔄 Benchmark 3: Feature Transformation Speed")
    transformer = FeatureTransformer()

    start_time = time.time()
    transformer.fit(feature_vectors)
    transformed = [transformer.transform(fv, method="minmax") for fv in feature_vectors]
    elapsed = time.time() - start_time

    transform_throughput = len(feature_vectors) / elapsed

    print(f"   ✓ Time: {elapsed:.2f}s")
    print(f"   ✓ Throughput: {transform_throughput:.2f} vectors/sec")

    # Benchmark 4: Feature Quality
    print("\n🎯 Benchmark 4: Feature Quality")

    # Completeness
    all_features = set()
    for fv in feature_vectors:
        all_features.update(fv.numerical_features.keys())

    completeness_scores = []
    for fv in feature_vectors:
        completeness = (len(fv.numerical_features) / len(all_features)) * 100
        completeness_scores.append(completeness)

    avg_completeness = np.mean(completeness_scores)
    print(f"   ✓ Average completeness: {avg_completeness:.1f}%")
    print(f"   ✓ Total unique features: {len(all_features)}")

    # Discriminability
    feature_names = sorted(feature_vectors[0].numerical_features.keys())
    X = np.array([fv.to_array(feature_names) for fv in feature_vectors])

    variances = np.var(X, axis=0)
    avg_variance = np.mean(variances)

    with np.errstate(divide='ignore', invalid='ignore'):
        cvs = np.std(X, axis=0) / np.abs(np.mean(X, axis=0))
        cvs = cvs[np.isfinite(cvs)]

    avg_cv = np.mean(cvs) if len(cvs) > 0 else 0
    discriminability_score = min(100, avg_cv * 10)

    print(f"   ✓ Discriminability score: {discriminability_score:.1f}/100")
    print(f"   ✓ Average CV: {avg_cv:.3f}")

    # Competitive Comparison
    print("\n🏆 Benchmark 5: Competitive Comparison")

    competitors = {
        "RChilli": {"feature_count": 200, "speed": 0.3},
        "Textkernel": {"feature_count": 250, "speed": 0.5},
        "Sovren": {"feature_count": 180, "speed": 0.4},
        "HireAbility": {"feature_count": 150, "speed": 0.6},
    }

    our_time_per_resume = elapsed / len(resumes)

    print("\n   Feature Count Comparison:")
    wins = 0
    for name, bench in competitors.items():
        comp_features = bench["feature_count"]
        improvement = ((avg_features - comp_features) / comp_features) * 100
        winner = "✅ Us" if avg_features >= comp_features else "❌ Them"
        if avg_features >= comp_features:
            wins += 1
        print(f"   vs {name}: {avg_features:.0f} vs {comp_features} ({improvement:+.1f}%) {winner}")

    print("\n   Processing Speed Comparison:")
    for name, bench in competitors.items():
        comp_time = bench["speed"]
        improvement = ((comp_time - our_time_per_resume) / comp_time) * 100
        winner = "✅ Us" if our_time_per_resume <= comp_time else "❌ Them"
        if our_time_per_resume <= comp_time:
            wins += 1
        print(f"   vs {name}: {our_time_per_resume:.3f}s vs {comp_time:.3f}s ({improvement:+.1f}%) {winner}")

    # Summary
    print("\n" + "="*80)
    print("📊 BENCHMARK SUMMARY")
    print("="*80)

    results = {
        "timestamp": datetime.now().isoformat(),
        "samples": n_samples,
        "metrics": {
            "extraction_speed": {
                "throughput": float(throughput),
                "unit": "resumes/sec"
            },
            "avg_features": {
                "value": float(avg_features),
                "unit": "features"
            },
            "completeness": {
                "value": float(avg_completeness),
                "unit": "%"
            },
            "discriminability": {
                "value": float(discriminability_score),
                "unit": "score"
            },
            "competitive_wins": {
                "value": wins,
                "total": len(competitors) * 2,
                "win_rate": (wins / (len(competitors) * 2)) * 100
            }
        }
    }

    # Save results
    output_file = "quick_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📈 Key Metrics:")
    print(f"   • Extraction Speed: {throughput:.2f} resumes/sec")
    print(f"   • Average Features: {avg_features:.0f}")
    print(f"   • Completeness: {avg_completeness:.1f}%")
    print(f"   • Discriminability: {discriminability_score:.1f}/100")
    print(f"   • Competitive Win Rate: {results['metrics']['competitive_wins']['win_rate']:.1f}%")

    print(f"\n💾 Results saved to: {output_file}")

    print("\n" + "="*80)
    print("✅ Benchmark completed successfully!")
    print("="*80)

    return results


if __name__ == "__main__":
    run_quick_benchmark()
