# Resume Screening System - Backend

Production-grade resume screening system using Process-Fit analysis (verb-based archetype profiling).

## Features

- **Process-Fit Analysis**: Analyzes resumes based on action verbs to determine process archetypes (Innovator, Leader, Maintainer, Problem-Solver, Enabler)
- **Semantic Matching**: Skills-based matching with fuzzy matching and synonym support
- **Hydrodynamic Control**: Optimizes compute costs by routing simple resumes to fast processing
- **Multi-language Support**: Handles both Hebrew and English text
- **RESTful API**: FastAPI-based API with automatic OpenAPI documentation
- **Database Persistence**: SQLAlchemy ORM with PostgreSQL/SQLite support
- **Email Notifications**: Mailgun integration for candidate and HR notifications
- **Webhook Support**: Integration endpoint for external systems (e.g., Opal)

## Architecture

```
backend/
├── models/          # Pydantic data models
├── engines/         # Core screening engines
│   ├── verb_extractor.py
│   ├── hydrodynamic.py
│   ├── semantic_matcher.py
│   └── screening_engine.py
├── database/        # SQLAlchemy setup and ORM models
├── routes/          # FastAPI route handlers
├── utils/           # Utilities (email, etc.)
├── config/          # Configuration and settings
└── main.py          # Application entry point
```

## Setup

### 1. Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:
- `DATABASE_URL`: Database connection string
- `MAILGUN_API_KEY`: Mailgun API key (optional)
- `MAILGUN_DOMAIN`: Mailgun domain (optional)
- `OPAL_WEBHOOK_SECRET`: Secret for Opal webhooks (optional)

### 3. Initialize Database

```bash
python -c "from database import init_db; init_db()"
```

### 4. Run the Server

```bash
# Development mode (with auto-reload)
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Jobs
- `POST /api/v1/jobs` - Create job requirement
- `GET /api/v1/jobs` - List all jobs
- `GET /api/v1/jobs/{id}` - Get specific job
- `PUT /api/v1/jobs/{id}` - Update job
- `DELETE /api/v1/jobs/{id}` - Delete job
- `GET /api/v1/jobs/{id}/stats` - Get job statistics

### Screening
- `POST /api/v1/screening/screen` - Screen a candidate
- `GET /api/v1/screening/{candidate_id}/result` - Get screening result

### Candidates
- `GET /api/v1/candidates` - List candidates
- `GET /api/v1/candidates/{id}` - Get candidate details
- `GET /api/v1/candidates/{id}/screening` - Get candidate screening result
- `GET /api/v1/candidates/search/by-decision` - Search by decision
- `GET /api/v1/candidates/search/by-archetype` - Search by archetype

### Webhooks
- `POST /api/v1/webhooks/opal` - Opal webhook endpoint

### Health
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/db` - Database health
- `GET /api/v1/health/info` - System info

## Usage Example

### 1. Create a Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Senior Backend Engineer",
    "archetype_primary": "Innovator",
    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
    "preferred_skills": ["Docker", "Kubernetes"]
  }'
```

### 2. Screen a Candidate

```bash
curl -X POST http://localhost:8000/api/v1/screening/screen \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_name": "John Doe",
    "candidate_email": "john@example.com",
    "resume_text": "Experienced engineer who created and developed...",
    "job_id": "<job_id_from_step_1>"
  }'
```

## Process Archetypes

The system identifies 5 process archetypes based on action verbs:

1. **Innovator** - Creates new systems, products, processes from scratch
   - Verbs: created, developed, built, innovated, pioneered
   - Context: "from scratch", "new", "prototype"

2. **Leader** - Leads teams and initiatives
   - Verbs: led, managed, directed, coached, influenced
   - Context: "team of", "cross-functional", "stakeholder"

3. **Maintainer** - Maintains and optimizes existing systems
   - Verbs: maintained, operated, optimized, improved
   - Context: "existing", "production", "SLA"

4. **Problem-Solver** - Diagnoses and solves complex problems
   - Verbs: solved, diagnosed, debugged, investigated
   - Context: "issue", "bug", "root cause"

5. **Enabler** - Supports and enables others' success
   - Verbs: supported, assisted, trained, mentored
   - Context: "enabled", "documentation", "knowledge sharing"

## Configuration

Key settings in `.env`:

- `PROCESS_FIT_WEIGHT`: Weight for process fit (default: 0.6)
- `SEMANTIC_FIT_WEIGHT`: Weight for semantic fit (default: 0.4)
- `PASS_THRESHOLD`: Minimum score to pass (default: 0.7)
- `ENABLE_HYDRODYNAMIC_CONTROL`: Enable cost optimization (default: true)
- `REYNOLDS_THRESHOLD_LOW`: Threshold for fast processing (default: 50)
- `REYNOLDS_THRESHOLD_HIGH`: Threshold for deep processing (default: 200)

## Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run tests
pytest

# With coverage
pytest --cov=. --cov-report=html
```

## License

Proprietary - All rights reserved
