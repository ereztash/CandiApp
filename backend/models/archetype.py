"""
Process Archetype Model
Defines the taxonomy of process-based archetypes used for resume screening.
"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass


class ProcessArchetype(str, Enum):
    """
    Process-based archetypes representing different work styles and approaches.
    Based on verb analysis of resume language.
    """
    INNOVATOR = "Innovator"
    LEADER = "Leader"
    MAINTAINER = "Maintainer"
    PROBLEM_SOLVER = "Problem-Solver"
    ENABLER = "Enabler"


@dataclass
class ArchetypeDefinition:
    """Definition of a process archetype with its linguistic markers."""
    name: ProcessArchetype
    hebrew_verbs: List[str]
    english_verbs: List[str]
    weight: float
    framenet_frames: List[str]
    context_markers: List[str]
    description: str


# Archetype Taxonomy
ARCHETYPE_TAXONOMY: Dict[ProcessArchetype, ArchetypeDefinition] = {
    ProcessArchetype.INNOVATOR: ArchetypeDefinition(
        name=ProcessArchetype.INNOVATOR,
        hebrew_verbs=[
            "יצר", "פיתח", "בנה", "חידש", "הקים", "השיק",
            "יזם", "תכנן", "עיצב", "הנדס", "המציא", "ייסד"
        ],
        english_verbs=[
            "created", "developed", "built", "innovated", "established",
            "launched", "pioneered", "designed", "conceived", "engineered",
            "invented", "founded", "architected", "initiated"
        ],
        weight=1.0,
        framenet_frames=[
            "Creating", "Intentionally_create", "Building",
            "Inventing", "Achieving_first"
        ],
        context_markers=[
            "from scratch", "new", "first-time", "prototype", "MVP",
            "הקמה", "חדש", "לראשונה", "אב טיפוס"
        ],
        description="Creates new systems, products, or processes from scratch. Focuses on innovation and first-time creation."
    ),

    ProcessArchetype.LEADER: ArchetypeDefinition(
        name=ProcessArchetype.LEADER,
        hebrew_verbs=[
            "הוביל", "ניהל", "הדריך", "השפיע", "איפשר",
            "העצים", "קידם", "יזם", "תיאם", "הנחה", "מנהל"
        ],
        english_verbs=[
            "led", "managed", "directed", "coached", "influenced",
            "enabled", "empowered", "orchestrated", "spearheaded",
            "championed", "drove", "mobilized", "guided"
        ],
        weight=0.95,
        framenet_frames=[
            "Leadership", "Influence", "Cause_to_start",
            "Assistance", "Subjective_influence"
        ],
        context_markers=[
            "team of", "cross-functional", "stakeholder", "initiative",
            "strategy", "צוות", "יוזמה", "אסטרטגיה"
        ],
        description="Leads teams and initiatives. Focuses on influencing, directing, and empowering others."
    ),

    ProcessArchetype.MAINTAINER: ArchetypeDefinition(
        name=ProcessArchetype.MAINTAINER,
        hebrew_verbs=[
            "תחזק", "שמר", "הפעיל", "אופטמז", "שיפר",
            "עקב", "ניטר", "הגביר", "ייעל", "תפעל"
        ],
        english_verbs=[
            "maintained", "operated", "optimized", "improved", "sustained",
            "monitored", "enhanced", "refined", "streamlined", "stabilized",
            "administered", "preserved"
        ],
        weight=0.85,
        framenet_frames=[
            "Preserving", "Maintaining", "Operating",
            "Sustaining", "Activity_ongoing"
        ],
        context_markers=[
            "existing", "day-to-day", "production", "SLA", "uptime",
            "קיים", "יומיום", "פרודקשן", "תפעול"
        ],
        description="Maintains and optimizes existing systems. Focuses on reliability, efficiency, and continuous improvement."
    ),

    ProcessArchetype.PROBLEM_SOLVER: ArchetypeDefinition(
        name=ProcessArchetype.PROBLEM_SOLVER,
        hebrew_verbs=[
            "פתר", "אבחן", "חקר", "זיהה", "חקר",
            "דיבג", "אישש", "ניתח", "גילה", "תיקן"
        ],
        english_verbs=[
            "solved", "diagnosed", "researched", "identified", "troubleshot",
            "debugged", "investigated", "analyzed", "discovered", "fixed",
            "resolved", "uncovered", "detected"
        ],
        weight=0.90,
        framenet_frames=[
            "Resolve_problem", "Finding", "Experimentation", "Research"
        ],
        context_markers=[
            "issue", "bug", "bottleneck", "root cause", "investigation",
            "בעיה", "באג", "צוואר בקבוק", "חקירה"
        ],
        description="Diagnoses and solves complex problems. Focuses on root cause analysis and troubleshooting."
    ),

    ProcessArchetype.ENABLER: ArchetypeDefinition(
        name=ProcessArchetype.ENABLER,
        hebrew_verbs=[
            "תמך", "סייע", "תיאם", "הנחה", "תרגל",
            "הדריך", "אימן", "שיתף", "העביר ידע", "תיעד"
        ],
        english_verbs=[
            "supported", "assisted", "coordinated", "facilitated", "trained",
            "guided", "collaborated", "mentored", "documented", "shared",
            "helped", "contributed", "partnered"
        ],
        weight=0.80,
        framenet_frames=[
            "Supporting", "Collaboration", "Assistance", "Communication"
        ],
        context_markers=[
            "enabled", "trained", "onboarded", "documentation", "mentoring",
            "תמיכה", "הדרכה", "תיעוד", "שיתוף פעולה"
        ],
        description="Supports and enables others' success. Focuses on collaboration, training, and knowledge sharing."
    )
}


@dataclass
class ArchetypeScore:
    """Score for a specific archetype with supporting evidence."""
    archetype: ProcessArchetype
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    evidence_verbs: List[str]
    evidence_count: int
    context_matches: List[str]


@dataclass
class ArchetypeProfile:
    """Complete archetype profile for a resume."""
    primary_archetype: ProcessArchetype
    primary_score: float
    primary_confidence: float
    secondary_archetype: Optional[ProcessArchetype]
    secondary_score: Optional[float]
    all_scores: Dict[ProcessArchetype, ArchetypeScore]
    total_verbs_found: int
    dominant_language: str  # "hebrew", "english", or "mixed"
