"""
SQLAlchemy model for the users table.
"""

from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship

from app.db.session import Base

class User(Base):
    """
    Reference to the users table in the main database.
    This is a read-only model to access user data.
    """
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Define relationships - these will be populated by back_populates in related models
    vehicles = relationship("Vehicle", back_populates="owner") 