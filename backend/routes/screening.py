"""
Screening API Routes
Main resume screening endpoints.
"""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
import logging

from database import get_db, JobDB, CandidateDB, ScreeningResultDB
from models.job import JobRequirement
from models.candidate import (
    ScreeningRequest,
    ScreeningResponse,
    ScreeningResult
)
from models.archetype import ProcessArchetype
from engines import ResumeScreeningEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/screening", tags=["screening"])

# Global screening engine instance
screening_engine = ResumeScreeningEngine()


@router.post("/screen", response_model=ScreeningResponse, status_code=status.HTTP_200_OK)
def screen_candidate(
    request: ScreeningRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> ScreeningResponse:
    """
    Screen a candidate's resume against job requirements.

    Args:
        request: Screening request with candidate and resume data
        background_tasks: FastAPI background tasks
        db: Database session

    Returns:
        ScreeningResponse with results

    Raises:
        HTTPException: If job not found or processing error
    """
    logger.info(
        f"Screening request for {request.candidate_email} "
        f"for job {request.job_id}"
    )

    # Validate job exists
    db_job = db.query(JobDB).filter(JobDB.id == str(request.job_id)).first()
    if not db_job:
        logger.error(f"Job not found: {request.job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {request.job_id} not found"
        )

    # Check if candidate already exists
    existing_candidate = db.query(CandidateDB).filter(
        CandidateDB.email == request.candidate_email,
        CandidateDB.job_id == str(request.job_id)
    ).first()

    if existing_candidate:
        logger.warning(
            f"Candidate {request.candidate_email} already screened "
            f"for job {request.job_id}"
        )
        # Return existing screening result
        existing_result = db.query(ScreeningResultDB).filter(
            ScreeningResultDB.candidate_id == existing_candidate.id
        ).first()

        if existing_result:
            return ScreeningResponse(
                success=True,
                screening_result=ScreeningResult.model_validate(existing_result),
                candidate_id=UUID(existing_candidate.id),
                message="Candidate already screened. Returning existing result."
            )

    # Create candidate record
    db_candidate = CandidateDB(
        name=request.candidate_name,
        email=request.candidate_email,
        resume_text=request.resume_text,
        job_id=str(request.job_id)
    )
    db.add(db_candidate)
    db.commit()
    db.refresh(db_candidate)

    logger.info(f"Candidate created: {db_candidate.id}")

    # Convert DB job to Pydantic model
    job_requirement = JobRequirement(
        id=UUID(db_job.id),
        name=db_job.name,
        description=db_job.description,
        archetype_primary=ProcessArchetype(db_job.archetype_primary),
        archetype_secondary=(
            ProcessArchetype(db_job.archetype_secondary)
            if db_job.archetype_secondary else None
        ),
        required_skills=db_job.required_skills or [],
        preferred_skills=db_job.preferred_skills or [],
        min_experience_years=db_job.min_experience_years,
        created_at=db_job.created_at,
        updated_at=db_job.updated_at,
        is_active=db_job.is_active,
        total_candidates_screened=db_job.total_candidates_screened
    )

    try:
        # Run screening
        screening_result = screening_engine.screen_resume(
            candidate_id=UUID(db_candidate.id),
            job_id=request.job_id,
            resume_text=request.resume_text,
            job=job_requirement
        )

        # Save screening result to database
        db_result = ScreeningResultDB(
            candidate_id=db_candidate.id,
            job_id=str(request.job_id),
            decision=screening_result.decision.value,
            overall_score=screening_result.overall_score,
            process_fit_score=screening_result.process_fit_score,
            semantic_fit_score=screening_result.semantic_fit_score,
            archetype_detected=screening_result.archetype_detected.value,
            archetype_confidence=screening_result.archetype_confidence,
            archetype_alignment=screening_result.archetype_alignment,
            evidence_verbs=screening_result.evidence_verbs,
            matched_skills=screening_result.matched_skills,
            missing_skills=screening_result.missing_skills,
            rejection_reason=(
                screening_result.rejection_reason.value
                if screening_result.rejection_reason else None
            ),
            rejection_details=screening_result.rejection_details,
            recommendation=screening_result.recommendation
        )

        db.add(db_result)

        # Update job screening count
        db_job.total_candidates_screened += 1

        db.commit()
        db.refresh(db_result)

        logger.info(
            f"Screening completed: {db_candidate.id} - "
            f"{screening_result.decision.value} ({screening_result.overall_score:.2%})"
        )

        # Schedule email sending in background
        # background_tasks.add_task(send_screening_email, screening_result, request.candidate_email)

        return ScreeningResponse(
            success=True,
            screening_result=screening_result,
            candidate_id=UUID(db_candidate.id),
            message="Candidate screened successfully"
        )

    except Exception as e:
        logger.error(f"Screening failed: {str(e)}", exc_info=True)
        # Clean up candidate if screening failed
        db.delete(db_candidate)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screening failed: {str(e)}"
        )


@router.get("/{candidate_id}/result", response_model=ScreeningResult)
def get_screening_result(
    candidate_id: UUID,
    db: Session = Depends(get_db)
) -> ScreeningResult:
    """
    Get screening result for a specific candidate.

    Args:
        candidate_id: Candidate UUID
        db: Database session

    Returns:
        ScreeningResult

    Raises:
        HTTPException: If result not found
    """
    result = db.query(ScreeningResultDB).filter(
        ScreeningResultDB.candidate_id == str(candidate_id)
    ).first()

    if not result:
        logger.warning(f"Screening result not found for candidate: {candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Screening result not found for candidate {candidate_id}"
        )

    return ScreeningResult.model_validate(result)
