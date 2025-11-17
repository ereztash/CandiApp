"""
Database Configuration
SQLAlchemy setup and session management.
"""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

from config.settings import settings

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    echo=settings.DEBUG  # Log SQL queries in debug mode
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session.

    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize database - create all tables.
    """
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")


def drop_db() -> None:
    """
    Drop all database tables.
    WARNING: This will delete all data!
    """
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.info("All tables dropped")
