"""
Main FastAPI Application
Entry point for the Resume Screening System API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from config.settings import settings
from database import init_db
from routes import jobs, screening, candidates, health, webhooks

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Resume Screening System API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Resume Screening System API...")


# Create FastAPI application
app = FastAPI(
    title="Resume Screening System API",
    description=(
        "Production-grade resume screening system using Process-Fit analysis. "
        "Analyzes resumes based on verb-based archetype profiling and semantic matching."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(jobs.router, prefix=settings.API_V1_PREFIX)
app.include_router(screening.router, prefix=settings.API_V1_PREFIX)
app.include_router(candidates.router, prefix=settings.API_V1_PREFIX)
app.include_router(webhooks.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root() -> dict:
    """
    Root endpoint.

    Returns:
        API information
    """
    return {
        "name": "Resume Screening System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "disabled",
        "api_prefix": settings.API_V1_PREFIX
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
