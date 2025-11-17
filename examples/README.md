# CandiApp Examples

This directory contains example scripts demonstrating how to use CandiApp.

## Available Examples

### 1. parse_resume.py

Parse a single resume and display extracted information.

```bash
python parse_resume.py <resume_file>
```

Example:
```bash
python parse_resume.py sample_resume.pdf
```

### 2. score_candidate.py

Score a candidate against job requirements.

```bash
python score_candidate.py <resume_file>
```

Example:
```bash
python score_candidate.py sample_resume.pdf
```

This example demonstrates:
- Parsing a resume
- Defining job requirements
- Scoring the candidate
- Getting match details and recommendations

## Sample Resumes

You can create your own sample resumes or use public resume examples for testing.

## Running Examples

All examples add the `src/` directory to the Python path, so you can run them directly without installing the package:

```bash
cd examples
python parse_resume.py your_resume.pdf
```
