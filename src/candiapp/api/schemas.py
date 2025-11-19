"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from enum import Enum


# Enums
class EducationLevelEnum(str, Enum):
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    CERTIFICATE = "certificate"
    OTHER = "other"


class ExperienceLevelEnum(str, Enum):
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


# Request Schemas
class JobRequirementsRequest(BaseModel):
    """Job requirements for scoring."""
    required_skills: List[str] = Field(default_factory=list, description="Required skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred skills")
    min_years_experience: Optional[int] = Field(None, ge=0, description="Minimum years of experience")
    max_years_experience: Optional[int] = Field(None, ge=0, description="Maximum years of experience")
    required_education: Optional[str] = Field(None, description="Required education level")
    industry: Optional[str] = Field(None, description="Industry")
    job_title: Optional[str] = Field(None, description="Job title")
    keywords: List[str] = Field(default_factory=list, description="Keywords to match")


class ScoreRequest(BaseModel):
    """Request to score a resume against job requirements."""
    resume_id: str = Field(..., description="Resume ID to score")
    job_requirements: JobRequirementsRequest = Field(..., description="Job requirements")


class BatchParseRequest(BaseModel):
    """Request for batch resume parsing."""
    resume_ids: List[str] = Field(..., description="List of resume IDs to parse")


class BatchScoreRequest(BaseModel):
    """Request for batch scoring."""
    resume_ids: List[str] = Field(..., description="List of resume IDs")
    job_requirements: JobRequirementsRequest = Field(..., description="Job requirements")


# Auth Schemas
class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, description="Full name")
    organization: Optional[str] = Field(None, description="Organization name")


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Expiration time in seconds")


class UserResponse(BaseModel):
    """User response."""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    full_name: Optional[str] = Field(None, description="Full name")
    organization: Optional[str] = Field(None, description="Organization")
    is_active: bool = Field(..., description="Is user active")
    created_at: datetime = Field(..., description="Creation timestamp")

    class Config:
        from_attributes = True


# Response Schemas
class ContactInfoResponse(BaseModel):
    """Contact information response."""
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


class EducationResponse(BaseModel):
    """Education entry response."""
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    gpa: Optional[float] = None
    level: Optional[str] = None
    description: Optional[str] = None


class ExperienceResponse(BaseModel):
    """Work experience response."""
    company: str
    title: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    current: bool = False
    location: Optional[str] = None
    description: Optional[str] = None
    responsibilities: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)


class SkillResponse(BaseModel):
    """Skill response."""
    name: str
    category: Optional[str] = None
    proficiency: Optional[str] = None
    years_of_experience: Optional[int] = None


class ParsedDataResponse(BaseModel):
    """Parsed resume data response."""
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    contact: ContactInfoResponse = Field(default_factory=ContactInfoResponse)
    summary: Optional[str] = None
    objective: Optional[str] = None
    experiences: List[ExperienceResponse] = Field(default_factory=list)
    total_experience_years: Optional[float] = None
    experience_level: Optional[str] = None
    education: List[EducationResponse] = Field(default_factory=list)
    highest_education: Optional[str] = None
    skills: List[SkillResponse] = Field(default_factory=list)
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    languages: List[Dict[str, Any]] = Field(default_factory=list)
    certifications: List[Dict[str, Any]] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    publications: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    parsing_confidence: Optional[float] = None


class ResumeResponse(BaseModel):
    """Complete resume response."""
    id: str = Field(..., description="Resume ID")
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="File type")
    file_size: int = Field(..., description="File size in bytes")
    parsed_data: Optional[ParsedDataResponse] = None
    parsing_time: Optional[float] = Field(None, description="Parsing time in seconds")
    parsing_errors: List[str] = Field(default_factory=list)
    created_at: datetime = Field(..., description="Upload timestamp")
    user_id: str = Field(..., description="Owner user ID")

    class Config:
        from_attributes = True


class ScoringResultResponse(BaseModel):
    """Scoring result response."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall score 0-100")
    dimension_scores: Dict[str, float] = Field(default_factory=dict, description="Dimension scores")
    match_details: Dict[str, Any] = Field(default_factory=dict, description="Match details")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    ranking: str = Field(..., description="Text ranking")


class ScoreResponse(BaseModel):
    """Score API response."""
    resume_id: str = Field(..., description="Resume ID")
    result: ScoringResultResponse = Field(..., description="Scoring result")
    scored_at: datetime = Field(..., description="Scoring timestamp")


class BatchParseResponse(BaseModel):
    """Batch parse response."""
    total: int = Field(..., description="Total resumes")
    successful: int = Field(..., description="Successfully parsed")
    failed: int = Field(..., description="Failed to parse")
    results: List[ResumeResponse] = Field(default_factory=list, description="Parse results")


class BatchScoreResponse(BaseModel):
    """Batch score response."""
    total: int = Field(..., description="Total resumes")
    results: List[ScoreResponse] = Field(default_factory=list, description="Score results")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Check timestamp")
    checks: Dict[str, bool] = Field(default_factory=dict, description="Component checks")


class FeaturesResponse(BaseModel):
    """Available features response."""
    version: str = Field(..., description="API version")
    total_features: int = Field(..., description="Total available features")
    features: Dict[str, bool] = Field(..., description="Feature availability")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[Dict[str, Any]] = Field(None, description="Error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List[Any] = Field(..., description="Result items")
    total: int = Field(..., description="Total items")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total pages")
