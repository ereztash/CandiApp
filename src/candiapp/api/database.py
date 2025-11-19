"""
Database models and connection management using SQLAlchemy.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID

from .config import settings

# Create engine
engine = create_engine(
    settings.database_url,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """Get database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_uuid():
    """Generate UUID string."""
    return str(uuid.uuid4())


class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    organization = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    resumes = relationship("ResumeRecord", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


class ResumeRecord(Base):
    """Resume storage model."""
    __tablename__ = "resumes"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=True)  # Storage path

    # Parsed data stored as JSON
    parsed_data = Column(JSON, nullable=True)
    parsing_time = Column(Float, nullable=True)
    parsing_errors = Column(JSON, default=list)
    parsing_confidence = Column(Float, nullable=True)

    # Extracted key fields for querying
    full_name = Column(String(255), nullable=True, index=True)
    email = Column(String(255), nullable=True, index=True)
    total_experience_years = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="resumes")
    scores = relationship("ScoreRecord", back_populates="resume", cascade="all, delete-orphan")


class JobRecord(Base):
    """Job description model."""
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Requirements as JSON
    requirements = Column(JSON, nullable=True)

    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ScoreRecord(Base):
    """Scoring result model."""
    __tablename__ = "scores"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    resume_id = Column(String(36), ForeignKey("resumes.id"), nullable=False, index=True)
    job_id = Column(String(36), nullable=True)  # Optional job reference

    # Scoring results
    overall_score = Column(Float, nullable=False)
    dimension_scores = Column(JSON, default=dict)
    match_details = Column(JSON, default=dict)
    recommendations = Column(JSON, default=list)
    ranking = Column(String(50), nullable=True)

    # Job requirements used for scoring
    job_requirements = Column(JSON, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    resume = relationship("ResumeRecord", back_populates="scores")


class APIKey(Base):
    """API key for programmatic access."""
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    key_hash = Column(String(255), nullable=False, unique=True)
    name = Column(String(100), nullable=False)

    # Permissions and limits
    is_active = Column(Boolean, default=True)
    rate_limit = Column(Integer, default=1000)  # Requests per day

    # Tracking
    last_used_at = Column(DateTime, nullable=True)
    total_requests = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")


class AuditLog(Base):
    """Audit log for tracking actions."""
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(36), nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)
