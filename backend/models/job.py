"""
Job Requirement Model
Defines job requirements including required archetypes and skills.
"""
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict

from models.archetype import ProcessArchetype


class JobRequirementBase(BaseModel):
    """Base schema for job requirements."""
    name: str = Field(..., min_length=1, max_length=255, description="Job title")
    description: Optional[str] = Field(None, description="Detailed job description")
    archetype_primary: ProcessArchetype = Field(..., description="Primary required archetype")
    archetype_secondary: Optional[ProcessArchetype] = Field(None, description="Secondary preferred archetype")
    required_skills: List[str] = Field(default_factory=list, description="Must-have skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills")
    min_experience_years: Optional[int] = Field(None, ge=0, le=50, description="Minimum years of experience")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Senior Backend Engineer",
                "description": "Building scalable microservices",
                "archetype_primary": "Innovator",
                "archetype_secondary": "Problem-Solver",
                "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
                "preferred_skills": ["Kubernetes", "AWS", "Redis"],
                "min_experience_years": 5
            }
        }
    )


class JobRequirementCreate(JobRequirementBase):
    """Schema for creating a new job requirement."""
    pass


class JobRequirementUpdate(BaseModel):
    """Schema for updating a job requirement."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    archetype_primary: Optional[ProcessArchetype] = None
    archetype_secondary: Optional[ProcessArchetype] = None
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    min_experience_years: Optional[int] = Field(None, ge=0, le=50)


class JobRequirement(JobRequirementBase):
    """Complete job requirement with metadata."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True, description="Whether the job is actively screening")
    total_candidates_screened: int = Field(default=0, description="Total candidates screened for this job")

    model_config = ConfigDict(from_attributes=True)


@dataclass
class ArchetypeFitCriteria:
    """Criteria for evaluating archetype fit for a job."""
    primary_archetype: ProcessArchetype
    secondary_archetype: Optional[ProcessArchetype]
    primary_weight: float = 0.7
    secondary_weight: float = 0.3
    min_primary_score: float = 0.6
    min_secondary_score: float = 0.4


@dataclass
class SkillMatchCriteria:
    """Criteria for evaluating skill match."""
    required_skills: List[str]
    preferred_skills: List[str]
    required_weight: float = 0.8
    preferred_weight: float = 0.2
    min_required_match_rate: float = 0.7  # Must match 70% of required skills


class JobWithStats(JobRequirement):
    """Job requirement with screening statistics."""
    total_screened: int = 0
    total_passed: int = 0
    total_failed: int = 0
    pass_rate: float = 0.0
    avg_overall_score: float = 0.0
    avg_process_fit: float = 0.0
    avg_semantic_fit: float = 0.0
    archetype_distribution: Dict[str, int] = Field(default_factory=dict)
