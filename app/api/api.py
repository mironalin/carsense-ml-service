from fastapi import APIRouter

from app.api.endpoints import health, models, predictions, auth

# Create API router
api_router = APIRouter()

# Include endpoint routers with appropriate prefixes and tags
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])