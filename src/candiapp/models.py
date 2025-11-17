"""
Data models for resume parsing and candidate information.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class EducationLevel(Enum):
    """Education level classifications."""
    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    CERTIFICATE = "certificate"
    OTHER = "other"


class ExperienceLevel(Enum):
    """Professional experience level classifications."""
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


@dataclass
class ContactInfo:
    """Contact information extracted from resume."""
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None


@dataclass
class Education:
    """Education entry."""
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    gpa: Optional[float] = None
    level: Optional[EducationLevel] = None
    description: Optional[str] = None


@dataclass
class Experience:
    """Work experience entry."""
    company: str
    title: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    current: bool = False
    location: Optional[str] = None
    description: Optional[str] = None
    responsibilities: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)


@dataclass
class Skill:
    """Skill with proficiency level."""
    name: str
    category: Optional[str] = None  # e.g., "Programming", "Language", "Tool"
    proficiency: Optional[str] = None  # e.g., "Beginner", "Intermediate", "Expert"
    years_of_experience: Optional[int] = None


@dataclass
class Certification:
    """Professional certification."""
    name: str
    issuer: Optional[str] = None
    issue_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    credential_id: Optional[str] = None


@dataclass
class Language:
    """Language proficiency."""
    language: str
    proficiency: Optional[str] = None  # e.g., "Native", "Fluent", "Professional", "Basic"


@dataclass
class Project:
    """Personal or professional project."""
    name: str
    description: Optional[str] = None
    technologies: List[str] = field(default_factory=list)
    url: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class ParsedData:
    """
    Complete parsed resume data structure.

    Designed to extract 200+ fields matching industry benchmarks from
    RChilli, Textkernel, and other leading parsers.
    """
    # Personal Information
    full_name: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    # Contact Information
    contact: ContactInfo = field(default_factory=ContactInfo)

    # Professional Summary
    summary: Optional[str] = None
    objective: Optional[str] = None

    # Experience
    experiences: List[Experience] = field(default_factory=list)
    total_experience_years: Optional[float] = None
    experience_level: Optional[ExperienceLevel] = None

    # Education
    education: List[Education] = field(default_factory=list)
    highest_education: Optional[EducationLevel] = None

    # Skills
    skills: List[Skill] = field(default_factory=list)
    technical_skills: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)

    # Languages
    languages: List[Language] = field(default_factory=list)

    # Certifications
    certifications: List[Certification] = field(default_factory=list)

    # Projects
    projects: List[Project] = field(default_factory=list)

    # Additional Information
    publications: List[str] = field(default_factory=list)
    awards: List[str] = field(default_factory=list)
    volunteer_experience: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    parsing_confidence: Optional[float] = None
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parsed data to dictionary."""
        return {
            "personal": {
                "full_name": self.full_name,
                "first_name": self.first_name,
                "last_name": self.last_name,
            },
            "contact": self.contact.__dict__,
            "summary": self.summary,
            "objective": self.objective,
            "experiences": [exp.__dict__ for exp in self.experiences],
            "total_experience_years": self.total_experience_years,
            "education": [edu.__dict__ for edu in self.education],
            "skills": [skill.__dict__ for skill in self.skills],
            "technical_skills": self.technical_skills,
            "soft_skills": self.soft_skills,
            "languages": [lang.__dict__ for lang in self.languages],
            "certifications": [cert.__dict__ for cert in self.certifications],
            "projects": [proj.__dict__ for proj in self.projects],
            "publications": self.publications,
            "awards": self.awards,
            "volunteer_experience": self.volunteer_experience,
            "metadata": {
                "parsing_confidence": self.parsing_confidence,
            }
        }


@dataclass
class Resume:
    """
    Complete resume object including file information and parsed data.
    """
    file_path: str
    file_type: str  # pdf, docx, doc, txt, html
    file_size: int
    parsed_data: Optional[ParsedData] = None
    parsing_time: Optional[float] = None  # seconds
    parsing_errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def is_parsed(self) -> bool:
        """Check if resume has been parsed."""
        return self.parsed_data is not None

    def get_field_count(self) -> int:
        """Get number of extracted fields (for benchmarking)."""
        if not self.parsed_data:
            return 0

        count = 0
        data_dict = self.parsed_data.to_dict()

        def count_fields(obj):
            nonlocal count
            if isinstance(obj, dict):
                for value in obj.values():
                    if value is not None:
                        count += 1
                        count_fields(value)
            elif isinstance(obj, list):
                count += len(obj)
                for item in obj:
                    count_fields(item)

        count_fields(data_dict)
        return count
