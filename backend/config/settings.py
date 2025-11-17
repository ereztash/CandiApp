"""
Application Settings
Manages configuration from environment variables.
"""
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./resume_screening.db",
        description="Database connection URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )

    # Email
    MAILGUN_API_KEY: Optional[str] = Field(default=None, description="Mailgun API key")
    MAILGUN_DOMAIN: Optional[str] = Field(default=None, description="Mailgun domain")
    SENDER_EMAIL: str = Field(
        default="noreply@company.com",
        description="Default sender email address"
    )

    # Opal Webhook
    OPAL_WEBHOOK_SECRET: Optional[str] = Field(
        default=None,
        description="Secret for validating Opal webhooks"
    )

    # Server
    ENVIRONMENT: str = Field(default="development", description="Environment name")
    DEBUG: bool = Field(default=True, description="Debug mode")
    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins"
    )

    # API
    API_PORT: int = Field(default=8000, description="API server port")
    API_HOST: str = Field(default="0.0.0.0", description="API server host")
    API_V1_PREFIX: str = Field(default="/api/v1", description="API v1 prefix")

    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-here-change-in-production",
        description="Secret key for JWT encoding"
    )
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )

    # Processing
    ENABLE_HYDRODYNAMIC_CONTROL: bool = Field(
        default=True,
        description="Enable hydrodynamic processing control"
    )
    PROCESS_FIT_WEIGHT: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for process fit in overall score"
    )
    SEMANTIC_FIT_WEIGHT: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for semantic fit in overall score"
    )
    PASS_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum overall score to pass screening"
    )

    # Hydrodynamic Control Thresholds
    REYNOLDS_THRESHOLD_LOW: float = Field(
        default=50.0,
        description="Below this, use Level 1 (deep) processing"
    )
    REYNOLDS_THRESHOLD_HIGH: float = Field(
        default=200.0,
        description="Above this, use Level 2 (fast) processing"
    )

    # Archetype Matching
    MIN_PRIMARY_ARCHETYPE_SCORE: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum score for primary archetype match"
    )
    MIN_REQUIRED_SKILLS_MATCH_RATE: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum percentage of required skills that must match"
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"
    )

    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse allowed origins into a list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"


# Global settings instance
settings = Settings()
