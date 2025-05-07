from datetime import datetime
from sqlalchemy import Column, DateTime, Integer
from sqlalchemy.ext.declarative import declared_attr

class BaseModel:
    """Base class for all database models."""

    # Generate tablename automatically
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()

    # Primary key with autoincrement=True to match PostgreSQL SERIAL type
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)