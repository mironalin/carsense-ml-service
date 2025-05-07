from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Health status of the API
    """
    health_status = {
        "status": "healthy",
        "api": "online",
    }

    return health_status