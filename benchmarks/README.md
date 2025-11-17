# CandiApp Benchmarking Suite

This directory contains tools and datasets for benchmarking the CandiApp resume parser against industry standards.

## Contents

- **generate_dataset.py** - Generate synthetic resume dataset with ground truth
- **run_benchmark.py** - Run parser benchmark and measure performance
- **BENCHMARK_REPORT.md** - Detailed benchmark analysis and results
- **benchmark_results.json** - Raw benchmark data (JSON format)
- **data/** - Generated datasets and test data

## Quick Start

### 1. Generate Dataset

```bash
python generate_dataset.py
```

This creates 50 synthetic resumes in `data/synthetic_resumes/` with ground truth labels.

### 2. Run Benchmark

```bash
python run_benchmark.py
```

This will:
- Parse all resumes in the dataset
- Measure speed, accuracy, and field extraction
- Compare against industry benchmarks
- Generate detailed results

### 3. View Results

- **Console:** Results printed to stdout
- **JSON:** `benchmark_results.json`
- **Report:** `BENCHMARK_REPORT.md`

## Benchmark Metrics

### Speed Performance
- **Target:** <1 second per resume (industry standard)
- **Measurement:** Average, median, min/max parsing time
- **Comparison:** Textkernel (300ms), RChilli (20ms)

### Accuracy
- **Target:** >90% field extraction accuracy
- **Measured Fields:**
  - Name (100% target)
  - Email (100% target)
  - Phone (90% target)
  - Skills (90% target)
  - Experience (80% target)
  - Education (80% target)

### Field Extraction
- **MVP Target:** 50+ fields
- **Production Target:** 100+ fields
- **Industry Leading:** 200+ fields

## Dataset Information

### Synthetic Resume Dataset
- **Size:** 50 resumes
- **Format:** TXT files
- **Language:** English (some Hebrew names)
- **Content:**
  - Professional summaries
  - Work experiences (2-5 per resume)
  - Education (1-2 per resume)
  - Technical skills (5-12 per resume)
  - Certifications (50% of resumes)
  - Contact information

### Ground Truth
Stored in `data/synthetic_resumes/ground_truth.json`:
- Name, email, phone
- Skills list
- Years of experience
- Experience entries (company, title, dates)
- Education entries (university, degree, year)
- Certifications
- Languages

## Latest Benchmark Results

**Date:** November 17, 2025
**Version:** CandiApp v0.1.0

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Parsing Speed | 0.86ms | <1000ms | ✅ PASS |
| Accuracy | 33.3% | >90% | ❌ FAIL |
| Fields | 19 | >50 | ❌ FAIL |
| Errors | 0% | <5% | ✅ PASS |

See **BENCHMARK_REPORT.md** for detailed analysis.

## Adding Custom Benchmarks

### Create Custom Dataset

1. Place resume files in `data/custom_resumes/`
2. Create `ground_truth.json` with expected values
3. Run: `python run_benchmark.py --data-dir data/custom_resumes`

### Test Against Real Resumes

```python
from candiapp import ResumeParser

parser = ResumeParser()
resume = parser.parse("path/to/resume.pdf")
print(f"Parsed in {resume.parsing_time:.3f}s")
print(f"Extracted {resume.get_field_count()} fields")
```

## Industry Benchmarks Reference

Based on research documented in `docs/benchmarks.md`:

- **Textkernel:** 300ms, 85-92% accuracy, 200+ fields
- **RChilli:** 20ms/CPU, 85-92% accuracy, 200+ fields
- **Fabric:** 1000+ resumes/min, 92% accuracy
- **Kickresume:** 92% hard skills detection

## Contributing

To add new benchmark tests:
1. Add test cases to `run_benchmark.py`
2. Update ground truth format if needed
3. Document new metrics in BENCHMARK_REPORT.md
4. Run full benchmark suite and verify results

## License

MIT License - Same as main project
