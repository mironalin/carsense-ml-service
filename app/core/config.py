from typing import List, Optional, Union
from pydantic import AnyHttpUrl, PostgresDsn, validator
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Explicitly load .env file and override existing environment variables
# This ensures that values from .env take precedence over system-wide environment variables.
load_dotenv(override=True)

class Settings(BaseSettings):
    """Base settings for the ML service."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CarSense ML Service"

    # CORS settings
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database settings
    # Default values for local development, override these in .env file
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "carsense_ml"
    POSTGRES_PORT: int = 5432

    # For Neon PostgreSQL, use the DATABASE_URL directly
    DATABASE_URL: Optional[str] = None

    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: dict) -> any:
        # If DATABASE_URL is provided (Neon), use it directly
        if values.get("DATABASE_URL"):
            return values.get("DATABASE_URL")

        # Otherwise, build the connection string from individual components
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            port=values.get("POSTGRES_PORT"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    # ML model settings
    MODEL_PATH: str = "models"
    MIN_PREDICTION_CONFIDENCE: float = 0.6

    # JWT Authentication settings - must match main backend
    JWT_SECRET_KEY: str = "your-secret-key"  # Change this in production
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Rate limiting settings
    RATE_LIMIT_PER_MINUTE: int = 60  # Default to 60 requests per minute
    ADMIN_RATE_LIMIT_PER_MINUTE: int = 300  # Higher limits for admins

    model_config = {
        "case_sensitive": True,
        "env_file": ".env"
    }

# Create settings instance
settings = Settings()