"""
Configuration management using Pydantic Settings.

Follows 12-factor app principles - all config from environment variables.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = Field(default="CandiApp", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: str = Field(default="development", description="Environment name")

    # API
    api_prefix: str = Field(default="/api/v1", description="API prefix")
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )

    # Database
    database_url: str = Field(
        default="postgresql://candiapp:candiapp@localhost:5432/candiapp",
        description="Database connection URL"
    )
    database_pool_size: int = Field(default=5, description="Database pool size")
    database_max_overflow: int = Field(default=10, description="Max overflow connections")

    # Redis
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # Authentication
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration in days"
    )

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Requests per window")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window")

    # File Upload
    max_upload_size_mb: int = Field(default=10, description="Max upload size in MB")
    allowed_file_types: List[str] = Field(
        default=["pdf", "docx", "doc", "txt"],
        description="Allowed file types"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format (json or text)")

    # Feature Flags
    enable_bert_features: bool = Field(
        default=False,
        description="Enable BERT features (requires GPU)"
    )
    enable_semantic_matching: bool = Field(
        default=True,
        description="Enable semantic skill matching"
    )
    enable_advanced_nlp: bool = Field(
        default=True,
        description="Enable advanced NLP features"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
