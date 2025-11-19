#!/usr/bin/env python3
"""
Run Complete Benchmark Suite

This script runs comprehensive benchmarks for the feature engineering system
and generates detailed reports.

Usage:
    python run_benchmarks.py --samples 500 --gpu
    python run_benchmarks.py --samples 1000 --no-gpu --output results.json
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark_framework import BenchmarkSuite
from synthetic_data_generator import SyntheticResumeGenerator
from report_generator import BenchmarkReporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive feature engineering benchmarks"
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
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results (default: benchmark_results_TIMESTAMP.json)"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Generate HTML report (default: True)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer samples, fewer iterations)"
    )

    return parser.parse_args()


def main():
    """Main benchmark execution."""
    args = parse_args()

    # Adjust samples for quick mode
    if args.quick:
        n_samples = min(args.samples, 100)
        logger.info("🏃 Running in QUICK mode")
    else:
        n_samples = args.samples

    logger.info("="*80)
    logger.info("🎯 CANDIAPP FEATURE ENGINEERING BENCHMARK SUITE")
    logger.info("="*80)
    logger.info(f"📊 Configuration:")
    logger.info(f"   - Samples: {n_samples}")
    logger.info(f"   - GPU: {'Enabled' if args.gpu else 'Disabled'}")
    logger.info(f"   - Mode: {'Quick' if args.quick else 'Full'}")
    logger.info("="*80)

    # Check GPU availability
    try:
        import torch
        if args.gpu and torch.cuda.is_available():
            logger.info(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        elif args.gpu:
            logger.warning("⚠️  GPU requested but not available, falling back to CPU")
            args.gpu = False
        else:
            logger.info("💻 Running on CPU (as requested)")
    except ImportError:
        logger.warning("⚠️  PyTorch not available, GPU benchmarks will be skipped")
        args.gpu = False

    # Step 1: Generate synthetic dataset
    logger.info("\n" + "="*80)
    logger.info("📝 STEP 1: Generating Synthetic Dataset")
    logger.info("="*80)

    from candiapp.models import ExperienceLevel

    generator = SyntheticResumeGenerator()

    # Distribution: More mid-level and senior candidates
    distribution = {
        ExperienceLevel.ENTRY: 0.10,
        ExperienceLevel.JUNIOR: 0.15,
        ExperienceLevel.MID: 0.30,
        ExperienceLevel.SENIOR: 0.25,
        ExperienceLevel.LEAD: 0.15,
        ExperienceLevel.EXECUTIVE: 0.05,
    }

    resumes, labels = generator.generate_dataset(
        n_samples=n_samples,
        distribution=distribution
    )

    logger.info(f"✅ Generated {len(resumes)} synthetic resumes")
    logger.info(f"   Distribution:")
    for level, count in zip(*np.unique(labels, return_counts=True)):
        level_name = list(ExperienceLevel)[level].value
        logger.info(f"   - {level_name}: {count} ({count/len(resumes)*100:.1f}%)")

    # Step 2: Run benchmarks
    logger.info("\n" + "="*80)
    logger.info("🚀 STEP 2: Running Benchmark Suite")
    logger.info("="*80)

    suite = BenchmarkSuite(use_gpu=args.gpu)
    results = suite.run_all(samples=resumes, labels=labels)

    # Step 3: Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"benchmark_results_{timestamp}.json"

    logger.info(f"\n💾 Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("✅ Results saved successfully")

    # Step 4: Generate report
    if args.report:
        logger.info("\n" + "="*80)
        logger.info("📊 STEP 3: Generating Report")
        logger.info("="*80)

        reporter = BenchmarkReporter(results)
        report_file = f"benchmark_report_{timestamp}.html"
        reporter.generate_html_report(report_file)

        logger.info(f"✅ HTML report generated: {report_file}")

        # Also print summary to console
        reporter.print_summary()

    # Step 5: Final summary
    logger.info("\n" + "="*80)
    logger.info("✨ BENCHMARK COMPLETED")
    logger.info("="*80)

    summary = results.get("summary", {})

    # Performance summary
    if "performance" in summary:
        logger.info("\n📈 Performance Highlights:")
        perf = summary["performance"]
        for name, data in perf.items():
            logger.info(f"   • {name}: {data['value']:.2f} {data['unit']}")

    # Quality summary
    if "quality" in summary:
        logger.info("\n🎯 Quality Highlights:")
        quality = summary["quality"]
        for name, data in quality.items():
            logger.info(f"   • {name}: {data['value']:.2f} {data['unit']}")

    # Competitive position
    if "competitive_position" in summary:
        logger.info("\n🏆 Competitive Position:")
        comp = summary["competitive_position"]
        logger.info(f"   • Wins: {comp['wins']}")
        logger.info(f"   • Ties: {comp['ties']}")
        logger.info(f"   • Losses: {comp['losses']}")
        logger.info(f"   • Win Rate: {comp['win_rate']:.1f}%")

    logger.info("\n" + "="*80)
    logger.info("🎉 All benchmarks completed successfully!")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
