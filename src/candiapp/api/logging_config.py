"""
Structured logging configuration.

Implements JSON logging for production and text logging for development.
"""

import logging
import sys
import json
import uuid
from datetime import datetime
from typing import Any, Dict
from contextvars import ContextVar

from .config import settings

# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            log_record["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_record["data"] = record.extra_data

        return json.dumps(log_record)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter for development."""

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        # Add request ID if available
        request_id = request_id_var.get()
        if request_id:
            record.msg = f"[{request_id[:8]}] {record.msg}"

        return super().format(record)


def setup_logging():
    """Configure application logging."""
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Remove existing handlers
    root_logger.handlers = []

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    # Set formatter based on config
    if settings.log_format.lower() == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    root_logger.addHandler(handler)

    # Set levels for noisy libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra context."""

    def process(self, msg, kwargs):
        # Add extra data to record
        extra = kwargs.get("extra", {})
        if self.extra:
            extra.update(self.extra)

        if extra:
            kwargs["extra"] = {"extra_data": extra}

        return msg, kwargs


def get_context_logger(name: str, **context) -> LoggerAdapter:
    """Get a logger with additional context."""
    logger = logging.getLogger(name)
    return LoggerAdapter(logger, context)


def set_request_id(request_id: str = None) -> str:
    """Set request ID for current context."""
    if not request_id:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def get_request_id() -> str:
    """Get current request ID."""
    return request_id_var.get()


# Initialize logging on module import
logger = setup_logging()
