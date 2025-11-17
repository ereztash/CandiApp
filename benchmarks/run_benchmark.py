#!/usr/bin/env python3
"""
Benchmark CandiApp Parser against synthetic dataset.

Measures:
- Parsing speed (time per resume)
- Accuracy (field extraction accuracy)
- Field count
- Error rate
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from candiapp import ResumeParser


@dataclass
class BenchmarkResult:
    """Results for a single resume benchmark."""
    resume_id: int
    parsing_time: float
    fields_extracted: int
    errors: List[str]
    accuracy: Dict[str, float]


def calculate_field_accuracy(parsed_data, ground_truth: Dict) -> Dict[str, float]:
    """Calculate accuracy for each field type."""
    accuracy = {}

    # Name accuracy
    if parsed_data.full_name:
        name_match = parsed_data.full_name.lower() == ground_truth['name'].lower()
        accuracy['name'] = 100.0 if name_match else 0.0
    else:
        accuracy['name'] = 0.0

    # Email accuracy
    if parsed_data.contact.email:
        email_match = parsed_data.contact.email.lower() == ground_truth['email'].lower()
        accuracy['email'] = 100.0 if email_match else 0.0
    else:
        accuracy['email'] = 0.0

    # Phone accuracy
    if parsed_data.contact.phone:
        # Normalize phone numbers for comparison
        parsed_phone = ''.join(c for c in parsed_data.contact.phone if c.isdigit())
        gt_phone = ''.join(c for c in ground_truth['phone'] if c.isdigit())
        phone_match = parsed_phone == gt_phone
        accuracy['phone'] = 100.0 if phone_match else 0.0
    else:
        accuracy['phone'] = 0.0

    # Skills accuracy
    if parsed_data.skills or parsed_data.technical_skills:
        parsed_skills = set()
        for skill in parsed_data.skills:
            parsed_skills.add(skill.name.lower())
        parsed_skills.update(s.lower() for s in parsed_data.technical_skills)

        gt_skills = set(s.lower() for s in ground_truth['skills'])

        if gt_skills:
            matched = len(parsed_skills & gt_skills)
            accuracy['skills'] = (matched / len(gt_skills)) * 100.0
        else:
            accuracy['skills'] = 100.0
    else:
        accuracy['skills'] = 0.0

    # Experience count
    if parsed_data.experiences:
        exp_diff = abs(len(parsed_data.experiences) - ground_truth['num_experiences'])
        accuracy['experience_count'] = max(0, 100 - (exp_diff * 25))
    else:
        accuracy['experience_count'] = 0.0

    # Education count
    if parsed_data.education:
        edu_diff = abs(len(parsed_data.education) - ground_truth['num_education'])
        accuracy['education_count'] = max(0, 100 - (edu_diff * 50))
    else:
        accuracy['education_count'] = 0.0

    return accuracy


def run_benchmark(data_dir: Path) -> List[BenchmarkResult]:
    """Run benchmark on all resumes in dataset."""
    # Load ground truth
    with open(data_dir / "ground_truth.json", 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)

    # Initialize parser
    parser = ResumeParser(enable_nlp=False)

    results = []
    resume_files = sorted(data_dir.glob("resume_*.txt"))

    print(f"Running benchmark on {len(resume_files)} resumes...")
    print("=" * 70)

    for i, resume_file in enumerate(resume_files):
        resume_id = int(resume_file.stem.split('_')[1])
        ground_truth = ground_truths[resume_id]

        # Parse resume
        start_time = time.time()
        resume = parser.parse(str(resume_file))
        parsing_time = time.time() - start_time

        # Calculate accuracy
        if resume.parsed_data:
            accuracy = calculate_field_accuracy(resume.parsed_data, ground_truth)
            fields_extracted = resume.get_field_count()
        else:
            accuracy = {k: 0.0 for k in ['name', 'email', 'phone', 'skills', 'experience_count', 'education_count']}
            fields_extracted = 0

        result = BenchmarkResult(
            resume_id=resume_id,
            parsing_time=parsing_time,
            fields_extracted=fields_extracted,
            errors=resume.parsing_errors,
            accuracy=accuracy
        )
        results.append(result)

        # Progress indicator
        if (i + 1) % 10 == 0:
            avg_time = statistics.mean([r.parsing_time for r in results])
            print(f"  Processed {i + 1}/{len(resume_files)} | Avg time: {avg_time:.3f}s")

    return results


def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Analyze benchmark results."""
    # Speed metrics
    parsing_times = [r.parsing_time for r in results]
    avg_time = statistics.mean(parsing_times)
    median_time = statistics.median(parsing_times)
    min_time = min(parsing_times)
    max_time = max(parsing_times)

    # Accuracy metrics
    field_accuracies = {
        'name': [],
        'email': [],
        'phone': [],
        'skills': [],
        'experience_count': [],
        'education_count': []
    }

    for result in results:
        for field, acc in result.accuracy.items():
            field_accuracies[field].append(acc)

    avg_accuracies = {
        field: statistics.mean(accs)
        for field, accs in field_accuracies.items()
    }
    overall_accuracy = statistics.mean(list(avg_accuracies.values()))

    # Field extraction
    fields_extracted = [r.fields_extracted for r in results]
    avg_fields = statistics.mean(fields_extracted)

    # Error rate
    error_count = sum(1 for r in results if r.errors)
    error_rate = (error_count / len(results)) * 100

    return {
        'speed': {
            'avg_time': avg_time,
            'median_time': median_time,
            'min_time': min_time,
            'max_time': max_time,
            'target': 1.0,  # Industry target: <1s
            'meets_target': avg_time < 1.0
        },
        'accuracy': {
            'overall': overall_accuracy,
            'by_field': avg_accuracies,
            'target': 90.0,  # Industry target: >90%
            'meets_target': overall_accuracy >= 90.0
        },
        'fields': {
            'avg_extracted': avg_fields,
            'target': 50,  # MVP target
            'meets_target': avg_fields >= 50
        },
        'errors': {
            'count': error_count,
            'rate': error_rate,
            'total_resumes': len(results)
        }
    }


def print_report(analysis: Dict[str, Any]):
    """Print benchmark report."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Speed metrics
    print("\n📊 PARSING SPEED")
    print("-" * 70)
    speed = analysis['speed']
    print(f"  Average Time:     {speed['avg_time']:.3f}s")
    print(f"  Median Time:      {speed['median_time']:.3f}s")
    print(f"  Min/Max Time:     {speed['min_time']:.3f}s / {speed['max_time']:.3f}s")
    print(f"  Industry Target:  <{speed['target']}s")
    print(f"  Status:           {'✓ PASS' if speed['meets_target'] else '✗ FAIL'}")

    # Accuracy metrics
    print("\n🎯 PARSING ACCURACY")
    print("-" * 70)
    acc = analysis['accuracy']
    print(f"  Overall Accuracy:  {acc['overall']:.1f}%")
    print(f"\n  By Field:")
    for field, value in acc['by_field'].items():
        field_name = field.replace('_', ' ').title()
        print(f"    {field_name:20s} {value:6.1f}%")
    print(f"\n  Industry Target:   >{acc['target']}%")
    print(f"  Status:            {'✓ PASS' if acc['meets_target'] else '✗ FAIL'}")

    # Field extraction
    print("\n📋 FIELD EXTRACTION")
    print("-" * 70)
    fields = analysis['fields']
    print(f"  Avg Fields:       {fields['avg_extracted']:.1f}")
    print(f"  MVP Target:       >={fields['target']}")
    print(f"  Status:           {'✓ PASS' if fields['meets_target'] else '✗ FAIL'}")

    # Errors
    print("\n⚠️  ERRORS")
    print("-" * 70)
    errors = analysis['errors']
    print(f"  Error Count:      {errors['count']}/{errors['total_resumes']}")
    print(f"  Error Rate:       {errors['rate']:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum([
        speed['meets_target'],
        acc['meets_target'],
        fields['meets_target']
    ])
    total = 3

    print(f"  Tests Passed:     {passed}/{total}")
    print(f"  Overall Status:   {'✓ PASS' if passed == total else '⚠ PARTIAL' if passed > 0 else '✗ FAIL'}")
    print("=" * 70)


def main():
    """Run benchmark and generate report."""
    data_dir = Path(__file__).parent / "data" / "synthetic_resumes"

    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print("Run generate_dataset.py first")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(data_dir)

    # Analyze results
    analysis = analyze_results(results)

    # Print report
    print_report(analysis)

    # Save detailed results
    output_file = Path(__file__).parent / "benchmark_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis': analysis,
            'results': [
                {
                    'id': r.resume_id,
                    'time': r.parsing_time,
                    'fields': r.fields_extracted,
                    'errors': r.errors,
                    'accuracy': r.accuracy
                }
                for r in results
            ]
        }, f, indent=2)

    print(f"\n💾 Detailed results saved to {output_file}")


if __name__ == "__main__":
    main()
