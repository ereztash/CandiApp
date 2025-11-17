# CandiApp

## AI-Powered Resume Parsing & Screening System

CandiApp is an intelligent resume parsing and candidate screening system built with industry-leading benchmarks in mind. The system analyzes, extracts, and scores resume data with high accuracy and speed.

## Purpose

This project implements a modern resume parsing system that:
- **Parses resumes** with high accuracy across multiple formats (PDF, DOCX, TXT)
- **Extracts structured data** including skills, experience, education, and contact information
- **Scores candidates** based on configurable criteria and job requirements
- **Benchmarks against industry standards** from leading ATS solutions (Textkernel, RChilli, Fabric)
- **Supports multiple languages** with focus on Hebrew and English

## Key Features

- **High-Performance Parsing**: Target processing speed <1 second per resume
- **200+ Data Fields**: Comprehensive extraction matching industry benchmarks
- **NLP-Powered Matching**: Context-aware skill and experience matching
- **Multi-Format Support**: PDF, DOCX, DOC, TXT, HTML resumes
- **Multi-Language**: Hebrew and English support out of the box
- **Configurable Scoring**: Customizable candidate ranking algorithms
- **Privacy-First**: GDPR-compliant data handling

## Industry Benchmarks

This project is designed to meet or exceed the following industry standards:

| Metric | Industry Standard | CandiApp Target |
|--------|------------------|-----------------|
| Parsing Accuracy | 85-92% | >90% |
| Processing Speed | 300ms-1s | <1s |
| Data Fields | 200+ | 200+ |
| Languages Supported | 20-40 | 2+ (Hebrew, English) |

See [docs/benchmarks.md](docs/benchmarks.md) for detailed benchmark analysis.

## Project Structure

```
CandiApp/
├── docs/              # Documentation and benchmarks
├── src/               # Source code
│   └── candiapp/     # Main package
│       ├── parser.py     # Resume parsing logic
│       ├── scoring.py    # Candidate scoring engine
│       └── models.py     # Data models
├── tests/            # Unit and integration tests
├── examples/         # Example usage and sample resumes
└── requirements.txt  # Python dependencies
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run example
python examples/parse_resume.py sample_resume.pdf
```

## Technology Stack

- **Python 3.9+**
- **spaCy**: NLP and entity recognition
- **PyPDF2/pdfplumber**: PDF parsing
- **python-docx**: DOCX parsing
- **scikit-learn**: ML-based scoring

## License

MIT License

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for details.
