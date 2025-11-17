"""
Webhook API Routes
Endpoints for external integrations (e.g., Opal).
"""
from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
import hmac
import hashlib
import logging
from typing import Optional

from database import get_db, JobDB
from models.candidate import ScreeningRequest
from routes.screening import screen_candidate
from config.settings import settings
from fastapi import BackgroundTasks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str
) -> bool:
    """
    Verify webhook signature using HMAC.

    Args:
        payload: Request payload
        signature: Provided signature
        secret: Shared secret

    Returns:
        True if signature is valid
    """
    expected_signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(signature, expected_signature)


@router.post("/opal")
async def opal_webhook(
    payload: dict,
    background_tasks: BackgroundTasks,
    x_opal_signature: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> dict:
    """
    Webhook endpoint for Opal integration.

    Expects payload:
    {
        "candidate_name": "John Doe",
        "candidate_email": "john@example.com",
        "resume_text": "...",
        "job_id": "uuid"
    }

    Args:
        payload: Webhook payload
        background_tasks: FastAPI background tasks
        x_opal_signature: Webhook signature header
        db: Database session

    Returns:
        Webhook processing result

    Raises:
        HTTPException: If validation fails
    """
    logger.info("Received Opal webhook")

    # Verify signature if secret is configured
    if settings.OPAL_WEBHOOK_SECRET and x_opal_signature:
        # In production, verify the actual payload bytes
        # For now, we'll skip signature verification
        logger.info("Webhook signature verification skipped (demo mode)")

    # Validate payload
    required_fields = ["candidate_name", "candidate_email", "resume_text", "job_id"]
    missing_fields = [f for f in required_fields if f not in payload]

    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required fields: {', '.join(missing_fields)}"
        )

    try:
        # Create screening request
        screening_request = ScreeningRequest(
            candidate_name=payload["candidate_name"],
            candidate_email=payload["candidate_email"],
            resume_text=payload["resume_text"],
            job_id=payload["job_id"]
        )

        # Process screening
        result = screen_candidate(
            request=screening_request,
            background_tasks=background_tasks,
            db=db
        )

        logger.info(
            f"Opal webhook processed successfully: {result.candidate_id}"
        )

        return {
            "success": True,
            "message": "Webhook processed successfully",
            "candidate_id": str(result.candidate_id),
            "decision": result.screening_result.decision.value,
            "overall_score": result.screening_result.overall_score
        }

    except Exception as e:
        logger.error(f"Webhook processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Webhook processing failed: {str(e)}"
        )
