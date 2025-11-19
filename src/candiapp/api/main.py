"""
Main FastAPI application.

CandiApp REST API for resume parsing and candidate scoring.
"""

import time
import tempfile
import os
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db, create_tables, ResumeRecord, ScoreRecord, User
from .auth import (
    get_current_user,
    get_current_user_or_api_key,
    authenticate_user,
    create_user,
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_api_key,
    hash_api_key,
)
from .schemas import (
    UserCreate,
    UserLogin,
    TokenResponse,
    UserResponse,
    JobRequirementsRequest,
    ScoreRequest,
    ResumeResponse,
    ParsedDataResponse,
    ScoreResponse,
    ScoringResultResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    FeaturesResponse,
    ErrorResponse,
)
from .logging_config import get_logger, set_request_id, get_request_id

# Import core candiapp modules
from ..parser import ResumeParser
from ..scoring import CandidateScorer, JobRequirements
from ..models import Resume
from .. import get_version_info

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting CandiApp API", extra={"version": settings.app_version})
    create_tables()
    yield
    # Shutdown
    logger.info("Shutting down CandiApp API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        description="AI-Powered Resume Parsing & Candidate Scoring API",
        version=settings.app_version,
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or set_request_id()
        set_request_id(request_id)

        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)

        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time": process_time,
            }
        )

        return response

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "message": exc.detail,
                "request_id": get_request_id(),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "request_id": get_request_id(),
            },
        )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI):
    """Register all API routes."""

    # Health check
    @app.get(
        f"{settings.api_prefix}/health",
        response_model=HealthResponse,
        tags=["Health"]
    )
    async def health_check(db: Session = Depends(get_db)):
        """Check API health status."""
        checks = {
            "api": True,
            "database": False,
        }

        # Check database
        try:
            db.execute("SELECT 1")
            checks["database"] = True
        except Exception:
            pass

        status_val = "healthy" if all(checks.values()) else "degraded"

        return HealthResponse(
            status=status_val,
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            checks=checks
        )

    # Features endpoint
    @app.get(
        f"{settings.api_prefix}/features",
        response_model=FeaturesResponse,
        tags=["Info"]
    )
    async def get_features():
        """Get available features and version info."""
        info = get_version_info()
        return FeaturesResponse(
            version=info["version"],
            total_features=info["feature_count"],
            features=info["features"]
        )

    # Auth endpoints
    @app.post(
        f"{settings.api_prefix}/auth/register",
        response_model=UserResponse,
        tags=["Authentication"]
    )
    async def register(user_data: UserCreate, db: Session = Depends(get_db)):
        """Register a new user."""
        user = create_user(
            db,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            organization=user_data.organization
        )
        logger.info(f"User registered: {user.email}")
        return user

    @app.post(
        f"{settings.api_prefix}/auth/login",
        response_model=TokenResponse,
        tags=["Authentication"]
    )
    async def login(user_data: UserLogin, db: Session = Depends(get_db)):
        """Login and get access token."""
        user = authenticate_user(db, user_data.email, user_data.password)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = create_access_token(data={"sub": user.id})
        refresh_token = create_refresh_token(data={"sub": user.id})

        logger.info(f"User logged in: {user.email}")

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.access_token_expire_minutes * 60
        )

    @app.post(
        f"{settings.api_prefix}/auth/refresh",
        response_model=TokenResponse,
        tags=["Authentication"]
    )
    async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
        """Refresh access token."""
        try:
            payload = decode_token(refresh_token)

            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )

            user_id = payload.get("sub")
            user = db.query(User).filter(User.id == user_id).first()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )

            access_token = create_access_token(data={"sub": user.id})
            new_refresh_token = create_refresh_token(data={"sub": user.id})

            return TokenResponse(
                access_token=access_token,
                refresh_token=new_refresh_token,
                expires_in=settings.access_token_expire_minutes * 60
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )

    @app.get(
        f"{settings.api_prefix}/auth/me",
        response_model=UserResponse,
        tags=["Authentication"]
    )
    async def get_me(current_user: User = Depends(get_current_user)):
        """Get current user info."""
        return current_user

    # Resume endpoints
    @app.post(
        f"{settings.api_prefix}/resumes/parse",
        response_model=ResumeResponse,
        tags=["Resumes"]
    )
    async def parse_resume(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user_or_api_key),
        db: Session = Depends(get_db)
    ):
        """Parse an uploaded resume."""
        # Validate file type
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in settings.allowed_file_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {settings.allowed_file_types}"
            )

        # Validate file size
        content = await file.read()
        file_size = len(content)
        max_size = settings.max_upload_size_mb * 1024 * 1024

        if file_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
            )

        # Save to temp file and parse
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # Parse resume
            start_time = time.time()
            parser = ResumeParser()
            resume = parser.parse(tmp_path)
            parsing_time = time.time() - start_time

            # Convert to dict for storage
            parsed_dict = resume.parsed_data.to_dict() if resume.parsed_data else None

            # Create database record
            resume_record = ResumeRecord(
                user_id=current_user.id,
                file_name=file.filename,
                file_type=file_ext,
                file_size=file_size,
                parsed_data=parsed_dict,
                parsing_time=parsing_time,
                parsing_errors=resume.parsing_errors,
                parsing_confidence=resume.parsed_data.parsing_confidence if resume.parsed_data else None,
                full_name=resume.parsed_data.full_name if resume.parsed_data else None,
                email=resume.parsed_data.contact.email if resume.parsed_data else None,
                total_experience_years=resume.parsed_data.total_experience_years if resume.parsed_data else None,
            )

            db.add(resume_record)
            db.commit()
            db.refresh(resume_record)

            logger.info(
                f"Resume parsed successfully",
                extra={
                    "resume_id": resume_record.id,
                    "user_id": current_user.id,
                    "parsing_time": parsing_time
                }
            )

            # Build response
            return ResumeResponse(
                id=resume_record.id,
                file_name=resume_record.file_name,
                file_type=resume_record.file_type,
                file_size=resume_record.file_size,
                parsed_data=parsed_dict,
                parsing_time=parsing_time,
                parsing_errors=resume_record.parsing_errors,
                created_at=resume_record.created_at,
                user_id=current_user.id
            )

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @app.get(
        f"{settings.api_prefix}/resumes",
        response_model=List[ResumeResponse],
        tags=["Resumes"]
    )
    async def list_resumes(
        skip: int = Query(0, ge=0),
        limit: int = Query(20, ge=1, le=100),
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """List user's parsed resumes."""
        resumes = db.query(ResumeRecord).filter(
            ResumeRecord.user_id == current_user.id
        ).offset(skip).limit(limit).all()

        return [
            ResumeResponse(
                id=r.id,
                file_name=r.file_name,
                file_type=r.file_type,
                file_size=r.file_size,
                parsed_data=r.parsed_data,
                parsing_time=r.parsing_time,
                parsing_errors=r.parsing_errors or [],
                created_at=r.created_at,
                user_id=r.user_id
            )
            for r in resumes
        ]

    @app.get(
        f"{settings.api_prefix}/resumes/{{resume_id}}",
        response_model=ResumeResponse,
        tags=["Resumes"]
    )
    async def get_resume(
        resume_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get a specific resume."""
        resume = db.query(ResumeRecord).filter(
            ResumeRecord.id == resume_id,
            ResumeRecord.user_id == current_user.id
        ).first()

        if not resume:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )

        return ResumeResponse(
            id=resume.id,
            file_name=resume.file_name,
            file_type=resume.file_type,
            file_size=resume.file_size,
            parsed_data=resume.parsed_data,
            parsing_time=resume.parsing_time,
            parsing_errors=resume.parsing_errors or [],
            created_at=resume.created_at,
            user_id=resume.user_id
        )

    @app.delete(
        f"{settings.api_prefix}/resumes/{{resume_id}}",
        tags=["Resumes"]
    )
    async def delete_resume(
        resume_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Delete a resume."""
        resume = db.query(ResumeRecord).filter(
            ResumeRecord.id == resume_id,
            ResumeRecord.user_id == current_user.id
        ).first()

        if not resume:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )

        db.delete(resume)
        db.commit()

        logger.info(f"Resume deleted: {resume_id}")

        return {"message": "Resume deleted successfully"}

    # Scoring endpoints
    @app.post(
        f"{settings.api_prefix}/score",
        response_model=ScoreResponse,
        tags=["Scoring"]
    )
    async def score_resume(
        request: ScoreRequest,
        current_user: User = Depends(get_current_user_or_api_key),
        db: Session = Depends(get_db)
    ):
        """Score a resume against job requirements."""
        # Get resume
        resume_record = db.query(ResumeRecord).filter(
            ResumeRecord.id == request.resume_id,
            ResumeRecord.user_id == current_user.id
        ).first()

        if not resume_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )

        if not resume_record.parsed_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume not parsed"
            )

        # Convert to Resume object for scoring
        from ..models import ParsedData, ContactInfo, Experience, Education, Skill

        parsed_data = ParsedData(
            full_name=resume_record.parsed_data.get("personal", {}).get("full_name"),
            summary=resume_record.parsed_data.get("summary"),
            total_experience_years=resume_record.parsed_data.get("total_experience_years"),
            technical_skills=resume_record.parsed_data.get("technical_skills", []),
        )

        # Add skills
        for skill_dict in resume_record.parsed_data.get("skills", []):
            parsed_data.skills.append(Skill(name=skill_dict.get("name", "")))

        # Add contact
        contact_dict = resume_record.parsed_data.get("contact", {})
        parsed_data.contact = ContactInfo(
            email=contact_dict.get("email"),
            phone=contact_dict.get("phone")
        )

        resume = Resume(
            file_path="",
            file_type=resume_record.file_type,
            file_size=resume_record.file_size,
            parsed_data=parsed_data
        )

        # Create job requirements
        job_req = JobRequirements(
            required_skills=request.job_requirements.required_skills,
            preferred_skills=request.job_requirements.preferred_skills,
            min_years_experience=request.job_requirements.min_years_experience,
            max_years_experience=request.job_requirements.max_years_experience,
            required_education=request.job_requirements.required_education,
            industry=request.job_requirements.industry,
            job_title=request.job_requirements.job_title,
            keywords=request.job_requirements.keywords,
        )

        # Score
        scorer = CandidateScorer(enable_semantic_matching=settings.enable_semantic_matching)
        result = scorer.score_candidate(resume, job_req)

        # Save score record
        score_record = ScoreRecord(
            resume_id=resume_record.id,
            overall_score=result.overall_score,
            dimension_scores=result.dimension_scores,
            match_details=result.match_details,
            recommendations=result.recommendations,
            ranking=result.ranking,
            job_requirements=request.job_requirements.model_dump()
        )

        db.add(score_record)
        db.commit()

        logger.info(
            f"Resume scored",
            extra={
                "resume_id": request.resume_id,
                "overall_score": result.overall_score
            }
        )

        return ScoreResponse(
            resume_id=request.resume_id,
            result=ScoringResultResponse(
                overall_score=result.overall_score,
                dimension_scores=result.dimension_scores,
                match_details=result.match_details,
                recommendations=result.recommendations,
                ranking=result.ranking
            ),
            scored_at=datetime.utcnow()
        )

    @app.post(
        f"{settings.api_prefix}/score/batch",
        response_model=BatchScoreResponse,
        tags=["Scoring"]
    )
    async def batch_score_resumes(
        request: BatchScoreRequest,
        current_user: User = Depends(get_current_user_or_api_key),
        db: Session = Depends(get_db)
    ):
        """Score multiple resumes against job requirements."""
        results = []

        for resume_id in request.resume_ids:
            try:
                score_request = ScoreRequest(
                    resume_id=resume_id,
                    job_requirements=request.job_requirements
                )
                result = await score_resume(score_request, current_user, db)
                results.append(result)
            except HTTPException as e:
                logger.warning(f"Failed to score resume {resume_id}: {e.detail}")

        return BatchScoreResponse(
            total=len(request.resume_ids),
            results=results
        )

    @app.get(
        f"{settings.api_prefix}/scores/{{resume_id}}",
        response_model=List[ScoreResponse],
        tags=["Scoring"]
    )
    async def get_resume_scores(
        resume_id: str,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
        """Get all scores for a resume."""
        # Verify ownership
        resume = db.query(ResumeRecord).filter(
            ResumeRecord.id == resume_id,
            ResumeRecord.user_id == current_user.id
        ).first()

        if not resume:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resume not found"
            )

        scores = db.query(ScoreRecord).filter(
            ScoreRecord.resume_id == resume_id
        ).order_by(ScoreRecord.created_at.desc()).all()

        return [
            ScoreResponse(
                resume_id=s.resume_id,
                result=ScoringResultResponse(
                    overall_score=s.overall_score,
                    dimension_scores=s.dimension_scores,
                    match_details=s.match_details,
                    recommendations=s.recommendations,
                    ranking=s.ranking
                ),
                scored_at=s.created_at
            )
            for s in scores
        ]


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "candiapp.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
