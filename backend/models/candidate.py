"""
Candidate and Screening Result Models
Defines candidate data and screening results.
"""
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, ConfigDict

from models.archetype import ProcessArchetype, ArchetypeProfile


class ScreeningDecision(str, Enum):
    """Final screening decision."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING_REVIEW = "PENDING_REVIEW"


class RejectionReason(str, Enum):
    """Standardized rejection reasons."""
    ARCHETYPE_MISMATCH = "ARCHETYPE_MISMATCH"
    INSUFFICIENT_SKILLS = "INSUFFICIENT_SKILLS"
    LOW_PROCESS_FIT = "LOW_PROCESS_FIT"
    LOW_SEMANTIC_FIT = "LOW_SEMANTIC_FIT"
    LOW_OVERALL_SCORE = "LOW_OVERALL_SCORE"
    INSUFFICIENT_EXPERIENCE = "INSUFFICIENT_EXPERIENCE"


class CandidateBase(BaseModel):
    """Base schema for candidate."""
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    resume_text: str = Field(..., min_length=10, description="Full resume text content")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "John Doe",
                "email": "john@example.com",
                "resume_text": "Experienced software engineer who developed and built multiple systems..."
            }
        }
    )


class CandidateCreate(CandidateBase):
    """Schema for creating a new candidate."""
    job_id: UUID = Field(..., description="Job being applied for")


class Candidate(CandidateBase):
    """Complete candidate with metadata."""
    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


@dataclass
class ProcessFitScore:
    """Process-fit scoring results."""
    overall_score: float  # 0.0 - 1.0
    archetype_profile: ArchetypeProfile
    archetype_match_score: float  # How well candidate matches required archetype
    archetype_alignment: str  # "perfect", "good", "partial", "mismatch"
    evidence_strength: float  # Quality of evidence (verb count, context matches)


@dataclass
class SemanticFitScore:
    """Semantic/skills-based scoring results."""
    overall_score: float  # 0.0 - 1.0
    matched_required_skills: List[str]
    matched_preferred_skills: List[str]
    missing_required_skills: List[str]
    missing_preferred_skills: List[str]
    required_match_rate: float  # % of required skills matched
    preferred_match_rate: float  # % of preferred skills matched
    additional_skills: List[str]  # Skills found but not in job requirements


class ScreeningResultBase(BaseModel):
    """Base screening result."""
    candidate_id: UUID
    job_id: UUID
    decision: ScreeningDecision
    overall_score: float = Field(..., ge=0.0, le=1.0)
    process_fit_score: float = Field(..., ge=0.0, le=1.0)
    semantic_fit_score: float = Field(..., ge=0.0, le=1.0)

    # Archetype Analysis
    archetype_detected: ProcessArchetype
    archetype_confidence: float = Field(..., ge=0.0, le=1.0)
    archetype_alignment: str
    evidence_verbs: List[str]

    # Skills Analysis
    matched_skills: List[str]
    missing_skills: List[str]

    # Decision Details
    rejection_reason: Optional[RejectionReason] = None
    rejection_details: Optional[str] = None
    recommendation: str  # Text recommendation for HR


class ScreeningResult(ScreeningResultBase):
    """Complete screening result with metadata."""
    id: UUID = Field(default_factory=uuid4)
    screened_at: datetime = Field(default_factory=datetime.utcnow)
    email_sent: bool = Field(default=False)
    hr_reviewed: bool = Field(default=False)
    hr_review_timestamp: Optional[datetime] = None
    hr_notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ScreeningRequest(BaseModel):
    """Request to screen a candidate."""
    candidate_name: str = Field(..., min_length=1, max_length=255)
    candidate_email: EmailStr
    resume_text: str = Field(..., min_length=10)
    job_id: UUID

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "candidate_name": "Jane Smith",
                "candidate_email": "jane@example.com",
                "resume_text": "Senior software engineer who led development of microservices...",
                "job_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }
    )


class ScreeningResponse(BaseModel):
    """Response from screening endpoint."""
    success: bool
    screening_result: ScreeningResult
    candidate_id: UUID
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "screening_result": {
                    "decision": "PASSED",
                    "overall_score": 0.85,
                    "archetype_detected": "Innovator",
                    "recommendation": "Strong candidate - proceed to interview"
                },
                "candidate_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "Candidate screened successfully"
            }
        }
    )


@dataclass
class ScreeningMetrics:
    """Metrics for screening performance."""
    total_screened: int
    total_passed: int
    total_failed: int
    pass_rate: float
    avg_processing_time_ms: float
    avg_overall_score: float
    archetype_distribution: Dict[ProcessArchetype, int]
