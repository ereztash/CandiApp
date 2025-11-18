# CandiApp API Reference

## Overview

CandiApp provides a RESTful API for resume parsing and candidate scoring. The API uses JSON for request/response bodies and JWT for authentication.

**Base URL:** `http://localhost:8000/api/v1`

**Interactive Documentation:**
- Swagger UI: `http://localhost:8000/api/v1/docs`
- ReDoc: `http://localhost:8000/api/v1/redoc`

---

## Authentication

All protected endpoints require authentication via JWT token or API key.

### Register User
```http
POST /auth/register
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "full_name": "John Doe",
  "organization": "Acme Corp"
}
```

**Response:**
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "full_name": "John Doe",
  "organization": "Acme Corp",
  "is_active": true,
  "created_at": "2025-01-01T00:00:00Z"
}
```

### Login
```http
POST /auth/login
```

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword123"
}
```

**Response:**
```json
{
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Using Authentication

**Bearer Token:**
```http
Authorization: Bearer <access_token>
```

**API Key:**
```http
X-API-Key: <api_key>
```

---

## Resume Endpoints

### Parse Resume
```http
POST /resumes/parse
Content-Type: multipart/form-data
```

**Request:**
- `file`: Resume file (PDF, DOCX, DOC, TXT)

**Response:**
```json
{
  "id": "uuid",
  "file_name": "resume.pdf",
  "file_type": "pdf",
  "file_size": 102400,
  "parsed_data": {
    "personal": {
      "full_name": "John Doe",
      "first_name": "John",
      "last_name": "Doe"
    },
    "contact": {
      "email": "john@example.com",
      "phone": "+1-555-123-4567",
      "linkedin": "linkedin.com/in/johndoe"
    },
    "summary": "Experienced software engineer...",
    "experiences": [...],
    "education": [...],
    "skills": [...],
    "technical_skills": ["Python", "JavaScript"],
    "soft_skills": ["Leadership", "Communication"]
  },
  "parsing_time": 0.234,
  "parsing_errors": [],
  "created_at": "2025-01-01T00:00:00Z",
  "user_id": "uuid"
}
```

### List Resumes
```http
GET /resumes?skip=0&limit=20
```

**Query Parameters:**
- `skip`: Number of items to skip (default: 0)
- `limit`: Number of items to return (default: 20, max: 100)

### Get Resume
```http
GET /resumes/{resume_id}
```

### Delete Resume
```http
DELETE /resumes/{resume_id}
```

---

## Scoring Endpoints

### Score Resume
```http
POST /score
```

**Request Body:**
```json
{
  "resume_id": "uuid",
  "job_requirements": {
    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
    "preferred_skills": ["Docker", "AWS", "Kubernetes"],
    "min_years_experience": 3,
    "max_years_experience": 10,
    "required_education": "bachelor",
    "industry": "Technology",
    "job_title": "Senior Software Engineer",
    "keywords": ["AI", "Machine Learning"]
  }
}
```

**Response:**
```json
{
  "resume_id": "uuid",
  "result": {
    "overall_score": 78.5,
    "dimension_scores": {
      "skills_match": 85.0,
      "experience_match": 90.0,
      "education_match": 75.0,
      "career_progression": 70.0,
      "recency": 100.0,
      "completeness": 80.0,
      "cultural_fit": 70.0
    },
    "match_details": {
      "matched_required_skills": ["Python", "FastAPI"],
      "missing_required_skills": ["PostgreSQL"],
      "total_experience_years": 5.0
    },
    "recommendations": [
      "Consider adding PostgreSQL to skills"
    ],
    "ranking": "Very Good Match"
  },
  "scored_at": "2025-01-01T00:00:00Z"
}
```

### Batch Score
```http
POST /score/batch
```

**Request Body:**
```json
{
  "resume_ids": ["uuid1", "uuid2", "uuid3"],
  "job_requirements": {
    "required_skills": ["Python"]
  }
}
```

### Get Resume Scores
```http
GET /scores/{resume_id}
```

Returns all historical scores for a resume.

---

## Health & Info Endpoints

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "timestamp": "2025-01-01T00:00:00Z",
  "checks": {
    "api": true,
    "database": true
  }
}
```

### Features
```http
GET /features
```

**Response:**
```json
{
  "version": "2.0.0",
  "total_features": 219,
  "features": {
    "advanced_features": true,
    "advanced_nlp": true,
    "semantic_matching": true,
    "bert": false,
    "enhanced_parser": true,
    "multilingual": true
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "detail": {},
  "request_id": "uuid"
}
```

### Common Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |

---

## Rate Limiting

- Default: 100 requests per minute
- Rate limit headers included in response:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

---

## Quick Start

### 1. Register and Login
```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

### 2. Parse a Resume
```bash
curl -X POST http://localhost:8000/api/v1/resumes/parse \
  -H "Authorization: Bearer <token>" \
  -F "file=@resume.pdf"
```

### 3. Score Against Job
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "resume_id": "<resume_id>",
    "job_requirements": {
      "required_skills": ["Python", "FastAPI"],
      "min_years_experience": 3
    }
  }'
```

---

## SDK Examples

### Python
```python
import httpx

# Login
response = httpx.post(
    "http://localhost:8000/api/v1/auth/login",
    json={"email": "user@example.com", "password": "pass123"}
)
token = response.json()["access_token"]

# Parse resume
with open("resume.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/resumes/parse",
        files={"file": f},
        headers={"Authorization": f"Bearer {token}"}
    )

resume = response.json()
print(f"Parsed: {resume['parsed_data']['personal']['full_name']}")

# Score
response = httpx.post(
    "http://localhost:8000/api/v1/score",
    json={
        "resume_id": resume["id"],
        "job_requirements": {
            "required_skills": ["Python"]
        }
    },
    headers={"Authorization": f"Bearer {token}"}
)

score = response.json()
print(f"Score: {score['result']['overall_score']}/100")
```

---

## Changelog

### v2.0.0
- Added REST API with FastAPI
- JWT and API key authentication
- PostgreSQL database support
- Structured JSON logging
- OpenAPI documentation
- GitHub Actions CI/CD

### v1.0.0
- Initial release
- Core parsing and scoring
