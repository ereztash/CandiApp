"""Models package for resume screening system."""
from models.archetype import (
    ProcessArchetype,
    ArchetypeDefinition,
    ArchetypeScore,
    ArchetypeProfile,
    ARCHETYPE_TAXONOMY
)
from models.job import (
    JobRequirement,
    JobRequirementCreate,
    JobRequirementUpdate,
    JobWithStats,
    ArchetypeFitCriteria,
    SkillMatchCriteria
)
from models.candidate import (
    Candidate,
    CandidateCreate,
    ScreeningResult,
    ScreeningRequest,
    ScreeningResponse,
    ScreeningDecision,
    RejectionReason,
    ProcessFitScore,
    SemanticFitScore,
    ScreeningMetrics
)

__all__ = [
    # Archetype
    "ProcessArchetype",
    "ArchetypeDefinition",
    "ArchetypeScore",
    "ArchetypeProfile",
    "ARCHETYPE_TAXONOMY",

    # Job
    "JobRequirement",
    "JobRequirementCreate",
    "JobRequirementUpdate",
    "JobWithStats",
    "ArchetypeFitCriteria",
    "SkillMatchCriteria",

    # Candidate
    "Candidate",
    "CandidateCreate",
    "ScreeningResult",
    "ScreeningRequest",
    "ScreeningResponse",
    "ScreeningDecision",
    "RejectionReason",
    "ProcessFitScore",
    "SemanticFitScore",
    "ScreeningMetrics",
]
