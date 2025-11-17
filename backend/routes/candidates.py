"""
Candidates API Routes
Endpoints for managing and viewing candidates.
"""
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging

from database import get_db, CandidateDB, ScreeningResultDB
from models.candidate import Candidate, ScreeningResult, ScreeningDecision
from models.archetype import ProcessArchetype

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/candidates", tags=["candidates"])


@router.get("/", response_model=List[Candidate])
def list_candidates(
    skip: int = 0,
    limit: int = 100,
    job_id: Optional[UUID] = None,
    db: Session = Depends(get_db)
) -> List[Candidate]:
    """
    List all candidates.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        job_id: Filter by job ID
        db: Database session

    Returns:
        List of candidates
    """
    query = db.query(CandidateDB)

    if job_id:
        query = query.filter(CandidateDB.job_id == str(job_id))

    candidates = query.offset(skip).limit(limit).all()

    logger.info(f"Retrieved {len(candidates)} candidates")

    return [Candidate.model_validate(c) for c in candidates]


@router.get("/{candidate_id}", response_model=Candidate)
def get_candidate(
    candidate_id: UUID,
    db: Session = Depends(get_db)
) -> Candidate:
    """
    Get a specific candidate by ID.

    Args:
        candidate_id: Candidate UUID
        db: Database session

    Returns:
        Candidate

    Raises:
        HTTPException: If candidate not found
    """
    candidate = db.query(CandidateDB).filter(
        CandidateDB.id == str(candidate_id)
    ).first()

    if not candidate:
        logger.warning(f"Candidate not found: {candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate with ID {candidate_id} not found"
        )

    return Candidate.model_validate(candidate)


@router.get("/{candidate_id}/screening", response_model=ScreeningResult)
def get_candidate_screening(
    candidate_id: UUID,
    db: Session = Depends(get_db)
) -> ScreeningResult:
    """
    Get screening result for a candidate.

    Args:
        candidate_id: Candidate UUID
        db: Database session

    Returns:
        ScreeningResult

    Raises:
        HTTPException: If screening result not found
    """
    result = db.query(ScreeningResultDB).filter(
        ScreeningResultDB.candidate_id == str(candidate_id)
    ).first()

    if not result:
        logger.warning(f"Screening result not found for: {candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screening result not found for candidate {candidate_id}"
        )

    return ScreeningResult.model_validate(result)


@router.get("/search/by-decision")
def search_by_decision(
    decision: ScreeningDecision = Query(..., description="Screening decision"),
    job_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[dict]:
    """
    Search candidates by screening decision.

    Args:
        decision: Screening decision (PASSED, FAILED, PENDING_REVIEW)
        job_id: Optional job ID filter
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of candidates with screening results
    """
    query = db.query(CandidateDB, ScreeningResultDB).join(
        ScreeningResultDB,
        CandidateDB.id == ScreeningResultDB.candidate_id
    ).filter(
        ScreeningResultDB.decision == decision.value
    )

    if job_id:
        query = query.filter(CandidateDB.job_id == str(job_id))

    results = query.offset(skip).limit(limit).all()

    logger.info(f"Found {len(results)} candidates with decision={decision.value}")

    return [
        {
            "candidate": Candidate.model_validate(candidate),
            "screening_result": ScreeningResult.model_validate(result)
        }
        for candidate, result in results
    ]


@router.get("/search/by-archetype")
def search_by_archetype(
    archetype: ProcessArchetype = Query(..., description="Process archetype"),
    job_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[dict]:
    """
    Search candidates by detected archetype.

    Args:
        archetype: Process archetype
        job_id: Optional job ID filter
        skip: Number of records to skip
        limit: Maximum number of records to return
        db: Database session

    Returns:
        List of candidates with screening results
    """
    query = db.query(CandidateDB, ScreeningResultDB).join(
        ScreeningResultDB,
        CandidateDB.id == ScreeningResultDB.candidate_id
    ).filter(
        ScreeningResultDB.archetype_detected == archetype.value
    )

    if job_id:
        query = query.filter(CandidateDB.job_id == str(job_id))

    results = query.offset(skip).limit(limit).all()

    logger.info(f"Found {len(results)} candidates with archetype={archetype.value}")

    return [
        {
            "candidate": Candidate.model_validate(candidate),
            "screening_result": ScreeningResult.model_validate(result)
        }
        for candidate, result in results
    ]
