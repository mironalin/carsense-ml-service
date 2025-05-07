from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Base settings for the ML service."""

    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CarSense ML Service"

    # ML model settings
    MODEL_PATH: str = "models"
    MIN_PREDICTION_CONFIDENCE: float = 0.6

    class Config:
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings()