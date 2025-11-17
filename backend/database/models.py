"""
Database ORM Models
SQLAlchemy models for database tables.
"""
from datetime import datetime
from uuid import uuid4
from sqlalchemy import (
    Column, String, Text, Float, Boolean, DateTime,
    Integer, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID as PostgreSQL_UUID
from sqlalchemy.orm import relationship
import uuid

from database.db import Base


# Custom UUID type that works with both PostgreSQL and SQLite
def generate_uuid():
    return str(uuid4())


class JobDB(Base):
    """Job requirements table."""
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    archetype_primary = Column(String(50), nullable=False, index=True)
    archetype_secondary = Column(String(50), nullable=True)
    required_skills = Column(JSON, nullable=False, default=list)
    preferred_skills = Column(JSON, nullable=False, default=list)
    min_experience_years = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    total_candidates_screened = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    candidates = relationship("CandidateDB", back_populates="job")
    screening_results = relationship("ScreeningResultDB", back_populates="job")

    def __repr__(self):
        return f"<JobDB(id={self.id}, name={self.name}, archetype={self.archetype_primary})>"


class CandidateDB(Base):
    """Candidates table."""
    __tablename__ = "candidates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    resume_text = Column(Text, nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    job = relationship("JobDB", back_populates="candidates")
    screening_result = relationship("ScreeningResultDB", back_populates="candidate", uselist=False)

    def __repr__(self):
        return f"<CandidateDB(id={self.id}, name={self.name}, email={self.email})>"


class ScreeningResultDB(Base):
    """Screening results table."""
    __tablename__ = "screening_results"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False, index=True, unique=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False, index=True)

    # Decision
    decision = Column(String(20), nullable=False, index=True)  # PASSED, FAILED, PENDING_REVIEW

    # Scores
    overall_score = Column(Float, nullable=False, index=True)
    process_fit_score = Column(Float, nullable=False)
    semantic_fit_score = Column(Float, nullable=False)

    # Archetype Analysis
    archetype_detected = Column(String(50), nullable=False, index=True)
    archetype_confidence = Column(Float, nullable=False)
    archetype_alignment = Column(String(20), nullable=False)  # perfect, good, partial, mismatch
    evidence_verbs = Column(JSON, nullable=False, default=list)

    # Skills Analysis
    matched_skills = Column(JSON, nullable=False, default=list)
    missing_skills = Column(JSON, nullable=False, default=list)

    # Rejection Details
    rejection_reason = Column(String(50), nullable=True)
    rejection_details = Column(Text, nullable=True)

    # Recommendation
    recommendation = Column(Text, nullable=False)

    # Status
    email_sent = Column(Boolean, default=False)
    hr_reviewed = Column(Boolean, default=False)
    hr_review_timestamp = Column(DateTime, nullable=True)
    hr_notes = Column(Text, nullable=True)

    # Timestamps
    screened_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    candidate = relationship("CandidateDB", back_populates="screening_result")
    job = relationship("JobDB", back_populates="screening_results")

    def __repr__(self):
        return (
            f"<ScreeningResultDB(id={self.id}, decision={self.decision}, "
            f"score={self.overall_score:.2f})>"
        )


class ScreeningHistoryDB(Base):
    """Screening history for analytics."""
    __tablename__ = "screening_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False, index=True)
    screening_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Scores
    overall_score = Column(Float, nullable=False)
    process_fit_score = Column(Float, nullable=False)

    # Decision
    decision = Column(String(20), nullable=False, index=True)
    reason = Column(Text, nullable=True)

    # Review
    hr_reviewed = Column(Boolean, default=False)
    hr_review_timestamp = Column(DateTime, nullable=True)

    def __repr__(self):
        return (
            f"<ScreeningHistoryDB(id={self.id}, decision={self.decision}, "
            f"timestamp={self.screening_timestamp})>"
        )
