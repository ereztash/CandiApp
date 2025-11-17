"""
Email Sender Utility
Handles sending email notifications via Mailgun.
"""
import logging
from typing import Optional
import requests

from config.settings import settings
from models.candidate import ScreeningResult, ScreeningDecision

logger = logging.getLogger(__name__)


class EmailSender:
    """Email sender using Mailgun API."""

    def __init__(self):
        """Initialize email sender."""
        self.api_key = settings.MAILGUN_API_KEY
        self.domain = settings.MAILGUN_DOMAIN
        self.sender_email = settings.SENDER_EMAIL
        self.enabled = bool(self.api_key and self.domain)

        if not self.enabled:
            logger.warning(
                "Email sending disabled - Mailgun credentials not configured"
            )

    def send_email(
        self,
        to_email: str,
        subject: str,
        text_body: str,
        html_body: Optional[str] = None
    ) -> bool:
        """
        Send an email via Mailgun.

        Args:
            to_email: Recipient email address
            subject: Email subject
            text_body: Plain text email body
            html_body: HTML email body (optional)

        Returns:
            True if email sent successfully
        """
        if not self.enabled:
            logger.warning(f"Email not sent (disabled): {subject} to {to_email}")
            return False

        try:
            response = requests.post(
                f"https://api.mailgun.net/v3/{self.domain}/messages",
                auth=("api", self.api_key),
                data={
                    "from": self.sender_email,
                    "to": to_email,
                    "subject": subject,
                    "text": text_body,
                    "html": html_body or text_body
                },
                timeout=10
            )

            if response.status_code == 200:
                logger.info(f"Email sent successfully to {to_email}")
                return True
            else:
                logger.error(
                    f"Failed to send email to {to_email}: "
                    f"{response.status_code} {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}", exc_info=True)
            return False

    def send_hr_notification(
        self,
        screening_result: ScreeningResult,
        candidate_name: str,
        job_name: str,
        ats_link: Optional[str] = None
    ) -> bool:
        """
        Send HR notification for passed candidates.

        Args:
            screening_result: Screening result
            candidate_name: Candidate name
            job_name: Job title
            ats_link: Optional ATS link

        Returns:
            True if email sent successfully
        """
        if screening_result.decision != ScreeningDecision.PASSED:
            return False

        subject = f"New Candidate: {candidate_name} - {job_name}"

        # Format evidence verbs
        evidence_str = ", ".join(screening_result.evidence_verbs[:5])
        if len(screening_result.evidence_verbs) > 5:
            evidence_str += f" (+{len(screening_result.evidence_verbs) - 5} more)"

        # Format matched skills
        matched_str = ", ".join(screening_result.matched_skills[:5])
        if len(screening_result.matched_skills) > 5:
            matched_str += f" (+{len(screening_result.matched_skills) - 5} more)"

        text_body = f"""
New Candidate: {candidate_name} - {job_name}
Process-Fit: {screening_result.overall_score:.0%}

Results:
- Overall Score: {screening_result.overall_score:.0%}
- Process Fit: {screening_result.process_fit_score:.0%}
- Semantic Fit: {screening_result.semantic_fit_score:.0%}
- Primary Archetype: {screening_result.archetype_detected.value} ({screening_result.archetype_confidence:.0%})
- Archetype Alignment: {screening_result.archetype_alignment}

Evidence Verbs: {evidence_str}
Matched Skills: {matched_str}

Recommendation: {screening_result.recommendation}

{f'ATS Link: {ats_link}' if ats_link else ''}
"""

        # TODO: Create HTML version
        return self.send_email(
            to_email=settings.SENDER_EMAIL,  # Send to configured HR email
            subject=subject,
            text_body=text_body
        )

    def send_candidate_rejection(
        self,
        candidate_email: str,
        candidate_name: str,
        screening_result: ScreeningResult,
        job_name: str
    ) -> bool:
        """
        Send rejection email to candidate with helpful feedback.

        Args:
            candidate_email: Candidate's email
            candidate_name: Candidate's name
            screening_result: Screening result
            job_name: Job title

        Returns:
            True if email sent successfully
        """
        if screening_result.decision != ScreeningDecision.FAILED:
            return False

        subject = f"Application Update - {job_name}"

        # Customize message based on rejection reason
        if screening_result.archetype_detected:
            found_archetype = screening_result.archetype_detected.value
            archetype_msg = (
                f"\n\nYour background aligns with a {found_archetype} profile. "
                f"This particular role requires different experience traits. "
                f"We encourage you to explore similar {found_archetype}-focused "
                f"positions that may be a better match."
            )
        else:
            archetype_msg = ""

        text_body = f"""
Hi {candidate_name},

Thank you for your interest in the {job_name} position.

After careful review of your application, we've decided to move forward with other candidates whose experience more closely aligns with the specific requirements for this role.{archetype_msg}

We appreciate the time you invested in the application process and wish you the best in your job search.

Best regards,
The Hiring Team
"""

        return self.send_email(
            to_email=candidate_email,
            subject=subject,
            text_body=text_body
        )

    def send_candidate_confirmation(
        self,
        candidate_email: str,
        candidate_name: str,
        job_name: str
    ) -> bool:
        """
        Send confirmation email to passed candidates.

        Args:
            candidate_email: Candidate's email
            candidate_name: Candidate's name
            job_name: Job title

        Returns:
            True if email sent successfully
        """
        subject = f"Application Update - {job_name}"

        text_body = f"""
Hi {candidate_name},

Thank you for your application for the {job_name} position.

We're pleased to inform you that your profile has passed our initial screening. Our HR team will review your application and will be in touch soon with next steps.

Best regards,
The Hiring Team
"""

        return self.send_email(
            to_email=candidate_email,
            subject=subject,
            text_body=text_body
        )


# Global email sender instance
email_sender = EmailSender()
