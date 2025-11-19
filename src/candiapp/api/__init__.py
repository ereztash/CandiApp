"""
CandiApp REST API Module

FastAPI-based REST API for resume parsing and candidate scoring.
"""

from .main import app, create_app
from .config import settings

__all__ = ["app", "create_app", "settings"]
