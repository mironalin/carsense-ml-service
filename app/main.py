import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sentry_sdk

from app.api.api import api_router
from app.core.config import settings
from app.core.middleware import add_middleware
# We'll import but not use it
from app.db.init_db import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Sentry if DSN is provided
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        # Enable performance monitoring
        enable_tracing=True,
        # You might want to set traces_sample_rate to 1.0 for production
        # or a lower value in high-traffic environments.
        # For development, 1.0 captures everything.
        traces_sample_rate=1.0,
        # You can also set profiles_sample_rate for profiling if needed
        # profiles_sample_rate=1.0,
        # Set environment (e.g., "development", "production")
        # environment="development", # Or load from another env var
    )
    logger.info(f"Sentry initialized with DSN: {settings.SENTRY_DSN[:settings.SENTRY_DSN.find('@')]}...") # Log only part of DSN
else:
    logger.info("Sentry DSN not found, skipping Sentry initialization.")

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="CarSense ML Service API for vehicle diagnostics and predictive maintenance",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
add_middleware(app)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "message": "Welcome to CarSense ML Service API",
        "version": "0.1.0",
        "docs_url": f"{settings.API_V1_STR}/docs",
    }

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Starting ML service...")
    logger.info("Connecting to database but skipping table creation")
    # We're not calling init_db() to avoid creating tables
    # The health endpoint will verify the database connection

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)