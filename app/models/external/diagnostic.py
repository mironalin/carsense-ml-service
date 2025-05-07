"""
SQLAlchemy models for diagnostic-related tables.
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base

class Diagnostic(Base):
    """
    Reference to the diagnostics table in the main database.
    """
    __tablename__ = "diagnostics"

    id = Column(Integer, primary_key=True)
    uuid = Column(UUID)
    vehicleId = Column(Integer, ForeignKey("vehicles.id"))
    timestamp = Column(DateTime)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)

    # Define relationships
    vehicle = relationship("Vehicle", back_populates="diagnostics")
    dtcCodes = relationship("DiagnosticDTC", back_populates="diagnostic")


class DiagnosticDTC(Base):
    """
    Reference to the diagnosticDTC table in the main database.
    """
    __tablename__ = "diagnosticDTC"

    id = Column(Integer, primary_key=True)
    uuid = Column(UUID)
    diagnosticId = Column(Integer, ForeignKey("diagnostics.id"))
    code = Column(String, ForeignKey("dtcLibrary.code"))
    confirmed = Column(Boolean)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)

    # Define relationships
    diagnostic = relationship("Diagnostic", back_populates="dtcCodes")
    dtcInfo = relationship("DTCLibrary", foreign_keys=[code])


class DTCLibrary(Base):
    """
    Reference to the dtcLibrary table in the main database.
    This is a read-only model to access DTC code information.
    """
    __tablename__ = "dtcLibrary"

    code = Column(String, primary_key=True)
    description = Column(Text, nullable=False)
    possibleCauses = Column(Text)
    severity = Column(String)
    components = Column(Text)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)