"""
Jobs API Routes
CRUD operations for job requirements.
"""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
import logging

from database import get_db, JobDB, ScreeningResultDB
from models.job import (
    JobRequirement,
    JobRequirementCreate,
    JobRequirementUpdate,
    JobWithStats
)
from models.candidate import ScreeningDecision

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/", response_model=JobRequirement, status_code=status.HTTP_201_CREATED)
def create_job(
    job: JobRequirementCreate,
    db: Session = Depends(get_db)
) -> JobRequirement:
    """
    Create a new job requirement.

    Args:
        job: Job requirement data
        db: Database session

    Returns:
        Created job requirement
    """
    logger.info(f"Creating new job: {job.name}")

    # Create database model
    db_job = JobDB(
        name=job.name,
        description=job.description,
        archetype_primary=job.archetype_primary.value,
        archetype_secondary=job.archetype_secondary.value if job.archetype_secondary else None,
        required_skills=job.required_skills,
        preferred_skills=job.preferred_skills,
        min_experience_years=job.min_experience_years
    )

    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    logger.info(f"Job created successfully: {db_job.id}")

    return JobRequirement.model_validate(db_job)


@router.get("/", response_model=List[JobRequirement])
def list_jobs(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    db: Session = Depends(get_db)
) -> List[JobRequirement]:
    """
    List all jobs.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        active_only: Only return active jobs
        db: Database session

    Returns:
        List of job requirements
    """
    query = db.query(JobDB)

    if active_only:
        query = query.filter(JobDB.is_active == True)

    jobs = query.offset(skip).limit(limit).all()

    logger.info(f"Retrieved {len(jobs)} jobs")

    return [JobRequirement.model_validate(job) for job in jobs]


@router.get("/{job_id}", response_model=JobRequirement)
def get_job(
    job_id: UUID,
    db: Session = Depends(get_db)
) -> JobRequirement:
    """
    Get a specific job by ID.

    Args:
        job_id: Job UUID
        db: Database session

    Returns:
        Job requirement

    Raises:
        HTTPException: If job not found
    """
    job = db.query(JobDB).filter(JobDB.id == str(job_id)).first()

    if not job:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )

    return JobRequirement.model_validate(job)


@router.put("/{job_id}", response_model=JobRequirement)
def update_job(
    job_id: UUID,
    job_update: JobRequirementUpdate,
    db: Session = Depends(get_db)
) -> JobRequirement:
    """
    Update a job requirement.

    Args:
        job_id: Job UUID
        job_update: Updated job data
        db: Database session

    Returns:
        Updated job requirement

    Raises:
        HTTPException: If job not found
    """
    db_job = db.query(JobDB).filter(JobDB.id == str(job_id)).first()

    if not db_job:
        logger.warning(f"Job not found for update: {job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )

    # Update fields
    update_data = job_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(db_job, field):
            # Convert enum to value if needed
            if hasattr(value, 'value'):
                value = value.value
            setattr(db_job, field, value)

    db.commit()
    db.refresh(db_job)

    logger.info(f"Job updated successfully: {job_id}")

    return JobRequirement.model_validate(db_job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(
    job_id: UUID,
    hard_delete: bool = False,
    db: Session = Depends(get_db)
) -> None:
    """
    Delete a job requirement.

    Args:
        job_id: Job UUID
        hard_delete: If True, permanently delete. If False, just mark as inactive.
        db: Database session

    Raises:
        HTTPException: If job not found
    """
    db_job = db.query(JobDB).filter(JobDB.id == str(job_id)).first()

    if not db_job:
        logger.warning(f"Job not found for deletion: {job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )

    if hard_delete:
        db.delete(db_job)
        logger.info(f"Job hard deleted: {job_id}")
    else:
        db_job.is_active = False
        logger.info(f"Job marked as inactive: {job_id}")

    db.commit()


@router.get("/{job_id}/stats", response_model=JobWithStats)
def get_job_stats(
    job_id: UUID,
    db: Session = Depends(get_db)
) -> JobWithStats:
    """
    Get job with screening statistics.

    Args:
        job_id: Job UUID
        db: Database session

    Returns:
        Job with statistics

    Raises:
        HTTPException: If job not found
    """
    job = db.query(JobDB).filter(JobDB.id == str(job_id)).first()

    if not job:
        logger.warning(f"Job not found for stats: {job_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )

    # Calculate statistics
    results = db.query(ScreeningResultDB).filter(
        ScreeningResultDB.job_id == str(job_id)
    ).all()

    total_screened = len(results)
    total_passed = sum(1 for r in results if r.decision == ScreeningDecision.PASSED.value)
    total_failed = total_screened - total_passed
    pass_rate = (total_passed / total_screened * 100) if total_screened > 0 else 0.0

    avg_overall_score = (
        sum(r.overall_score for r in results) / total_screened
        if total_screened > 0 else 0.0
    )
    avg_process_fit = (
        sum(r.process_fit_score for r in results) / total_screened
        if total_screened > 0 else 0.0
    )
    avg_semantic_fit = (
        sum(r.semantic_fit_score for r in results) / total_screened
        if total_screened > 0 else 0.0
    )

    # Archetype distribution
    archetype_dist = {}
    for result in results:
        archetype = result.archetype_detected
        archetype_dist[archetype] = archetype_dist.get(archetype, 0) + 1

    # Create response
    job_dict = {
        **JobRequirement.model_validate(job).model_dump(),
        "total_screened": total_screened,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "pass_rate": pass_rate,
        "avg_overall_score": avg_overall_score,
        "avg_process_fit": avg_process_fit,
        "avg_semantic_fit": avg_semantic_fit,
        "archetype_distribution": archetype_dist
    }

    return JobWithStats(**job_dict)
