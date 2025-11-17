"""
CandiApp - AI-Powered Resume Parsing & Screening System

A modern resume parsing and candidate screening system with industry-leading benchmarks.
"""

__version__ = "0.1.0"
__author__ = "CandiApp Team"

from .parser import ResumeParser
from .models import Resume, ParsedData
from .scoring import CandidateScorer

__all__ = [
    "ResumeParser",
    "Resume",
    "ParsedData",
    "CandidateScorer",
]
