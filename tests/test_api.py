"""
Comprehensive API tests for CandiApp REST API.

Tests cover authentication, resume parsing, scoring, and error handling.
"""

import pytest
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Use SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def test_db():
    """Create test database."""
    from candiapp.api.database import Base

    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    yield TestingSessionLocal()

    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db):
    """Create test client with database override."""
    from candiapp.api.main import app
    from candiapp.api.database import get_db

    def override_get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    """Create authenticated user and return headers."""
    # Register user
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "testpassword123",
            "full_name": "Test User"
        }
    )
    assert response.status_code == 200

    # Login
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "testpassword123"
        }
    )
    assert response.status_code == 200

    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "checks" in data

    def test_features_endpoint(self, client):
        """Test features endpoint."""
        response = client.get("/api/v1/features")
        assert response.status_code == 200

        data = response.json()
        assert "version" in data
        assert "total_features" in data
        assert "features" in data
        assert data["total_features"] >= 127


class TestAuthentication:
    """Tests for authentication endpoints."""

    def test_register_user(self, client):
        """Test user registration."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "full_name": "New User",
                "organization": "Test Org"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert "id" in data
        assert data["is_active"] == True

    def test_register_duplicate_email(self, client):
        """Test registration with duplicate email."""
        # First registration
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "password123"
            }
        )

        # Second registration with same email
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "duplicate@example.com",
                "password": "password456"
            }
        )
        assert response.status_code == 400
        assert "already registered" in response.json()["message"]

    def test_register_invalid_email(self, client):
        """Test registration with invalid email."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "password": "password123"
            }
        )
        assert response.status_code == 422

    def test_register_short_password(self, client):
        """Test registration with short password."""
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "user@example.com",
                "password": "short"
            }
        )
        assert response.status_code == 422

    def test_login_success(self, client):
        """Test successful login."""
        # Register
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "login@example.com",
                "password": "password123"
            }
        )

        # Login
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "login@example.com",
                "password": "password123"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data

    def test_login_wrong_password(self, client):
        """Test login with wrong password."""
        # Register
        client.post(
            "/api/v1/auth/register",
            json={
                "email": "user@example.com",
                "password": "correctpassword"
            }
        )

        # Login with wrong password
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "user@example.com",
                "password": "wrongpassword"
            }
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent user."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "password123"
            }
        )
        assert response.status_code == 401

    def test_get_current_user(self, client, auth_headers):
        """Test get current user endpoint."""
        response = client.get("/api/v1/auth/me", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"

    def test_get_current_user_unauthorized(self, client):
        """Test get current user without auth."""
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


class TestResumeEndpoints:
    """Tests for resume parsing endpoints."""

    def test_parse_resume_success(self, client, auth_headers):
        """Test successful resume parsing."""
        # Create a simple text resume
        resume_content = """
        John Doe
        Email: john.doe@example.com
        Phone: +1-555-123-4567

        Summary
        Experienced software engineer with 5 years of experience.

        Skills
        Python, JavaScript, React, Node.js

        Experience
        Senior Developer at Tech Corp
        2020 - Present

        Education
        Bachelor of Science in Computer Science
        MIT, 2015
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(resume_content)
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/api/v1/resumes/parse",
                    files={"file": ("resume.txt", f, "text/plain")},
                    headers=auth_headers
                )

            assert response.status_code == 200

            data = response.json()
            assert "id" in data
            assert data["file_name"] == "resume.txt"
            assert data["file_type"] == "txt"
            assert "parsed_data" in data
            assert "parsing_time" in data
        finally:
            os.unlink(temp_path)

    def test_parse_resume_unauthorized(self, client):
        """Test resume parsing without auth."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test resume")
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/api/v1/resumes/parse",
                    files={"file": ("resume.txt", f, "text/plain")}
                )

            assert response.status_code == 401
        finally:
            os.unlink(temp_path)

    def test_parse_resume_invalid_file_type(self, client, auth_headers):
        """Test resume parsing with invalid file type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
            f.write("Test")
            temp_path = f.name

        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/api/v1/resumes/parse",
                    files={"file": ("malware.exe", f, "application/octet-stream")},
                    headers=auth_headers
                )

            assert response.status_code == 400
            assert "not allowed" in response.json()["message"]
        finally:
            os.unlink(temp_path)

    def test_list_resumes(self, client, auth_headers):
        """Test listing resumes."""
        response = client.get("/api/v1/resumes", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_resumes_pagination(self, client, auth_headers):
        """Test resume list pagination."""
        response = client.get(
            "/api/v1/resumes?skip=0&limit=10",
            headers=auth_headers
        )
        assert response.status_code == 200

    def test_get_resume_not_found(self, client, auth_headers):
        """Test getting nonexistent resume."""
        response = client.get(
            "/api/v1/resumes/nonexistent-id",
            headers=auth_headers
        )
        assert response.status_code == 404

    def test_delete_resume_not_found(self, client, auth_headers):
        """Test deleting nonexistent resume."""
        response = client.delete(
            "/api/v1/resumes/nonexistent-id",
            headers=auth_headers
        )
        assert response.status_code == 404


class TestScoringEndpoints:
    """Tests for candidate scoring endpoints."""

    def test_score_resume_not_found(self, client, auth_headers):
        """Test scoring nonexistent resume."""
        response = client.post(
            "/api/v1/score",
            json={
                "resume_id": "nonexistent-id",
                "job_requirements": {
                    "required_skills": ["Python"],
                    "min_years_experience": 2
                }
            },
            headers=auth_headers
        )
        assert response.status_code == 404

    def test_score_request_validation(self, client, auth_headers):
        """Test score request validation."""
        # Missing resume_id
        response = client.post(
            "/api/v1/score",
            json={
                "job_requirements": {
                    "required_skills": ["Python"]
                }
            },
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_batch_score_empty_list(self, client, auth_headers):
        """Test batch scoring with empty list."""
        response = client.post(
            "/api/v1/score/batch",
            json={
                "resume_ids": [],
                "job_requirements": {
                    "required_skills": ["Python"]
                }
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["total"] == 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_error(self, client):
        """Test 404 error response."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.patch("/api/v1/health")
        assert response.status_code == 405

    def test_request_id_header(self, client):
        """Test request ID in response headers."""
        response = client.get("/api/v1/health")
        assert "X-Request-ID" in response.headers


class TestJobRequirementsValidation:
    """Tests for job requirements schema validation."""

    def test_valid_job_requirements(self, client, auth_headers):
        """Test valid job requirements."""
        job_req = {
            "required_skills": ["Python", "FastAPI"],
            "preferred_skills": ["Docker", "AWS"],
            "min_years_experience": 3,
            "max_years_experience": 10,
            "required_education": "bachelor",
            "industry": "Technology",
            "job_title": "Senior Software Engineer",
            "keywords": ["AI", "ML"]
        }

        # Just validate it can be used in a request
        response = client.post(
            "/api/v1/score",
            json={
                "resume_id": "test-id",
                "job_requirements": job_req
            },
            headers=auth_headers
        )
        # 404 is expected since resume doesn't exist
        assert response.status_code == 404

    def test_invalid_experience_values(self, client, auth_headers):
        """Test negative experience values."""
        response = client.post(
            "/api/v1/score",
            json={
                "resume_id": "test-id",
                "job_requirements": {
                    "min_years_experience": -1
                }
            },
            headers=auth_headers
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
