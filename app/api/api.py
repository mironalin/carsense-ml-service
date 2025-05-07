from fastapi import APIRouter

from app.api.endpoints import health

# Create API router
api_router = APIRouter()

# Include endpoint routers with appropriate prefixes and tags
api_router.include_router(health.router, prefix="/health", tags=["health"])