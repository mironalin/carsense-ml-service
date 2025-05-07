from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import text

from app.db.session import get_db

router = APIRouter()

@router.get("/")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint that verifies API and database status.

    Args:
        db: Database session dependency

    Returns:
        dict: Health status of the API and database
    """
    health_status = {
        "status": "healthy",
        "api": "online",
        "database": "online",
    }

    # Check database connection
    try:
        # Use text() for raw SQL queries in SQLAlchemy 2.0+
        db.execute(text("SELECT 1"))
    except SQLAlchemyError as e:
        health_status["database"] = "offline"
        health_status["status"] = "unhealthy"
        health_status["database_error"] = str(e)

    return health_status