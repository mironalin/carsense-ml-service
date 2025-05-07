from fastapi import APIRouter

from app.api.endpoints import health, models, predictions

# Create API router
api_router = APIRouter()

# Include endpoint routers with appropriate prefixes and tags
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])