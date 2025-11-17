"""
Health Check API Routes
System health and status endpoints.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
from datetime import datetime

from database import get_db
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
def health_check() -> dict:
    """
    Basic health check endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ENVIRONMENT
    }


@router.get("/db")
def database_health(db: Session = Depends(get_db)) -> dict:
    """
    Check database connectivity.

    Args:
        db: Database session

    Returns:
        Database health status
    """
    try:
        # Execute a simple query
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/info")
def system_info() -> dict:
    """
    Get system information and configuration.

    Returns:
        System information
    """
    return {
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "api_version": "v1",
        "features": {
            "hydrodynamic_control": settings.ENABLE_HYDRODYNAMIC_CONTROL,
            "process_fit_weight": settings.PROCESS_FIT_WEIGHT,
            "semantic_fit_weight": settings.SEMANTIC_FIT_WEIGHT,
            "pass_threshold": settings.PASS_THRESHOLD
        },
        "timestamp": datetime.utcnow().isoformat()
    }
