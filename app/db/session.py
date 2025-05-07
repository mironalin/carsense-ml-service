from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Create database engine - convert PostgresDsn to string
db_url = str(settings.SQLALCHEMY_DATABASE_URI)
engine = create_engine(
    db_url,
    pool_pre_ping=True  # Test connections for liveness when checked out from pool
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for declarative class definitions
Base = declarative_base()

# Dependency to get database session
def get_db():
    """
    Dependency for FastAPI endpoints that need a database session.
    Creates a new session for each request and closes it when done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()