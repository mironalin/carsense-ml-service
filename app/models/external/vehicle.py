"""
SQLAlchemy model for the vehicles table.
"""

from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base

class Vehicle(Base):
    """
    Reference to the vehicles table in the main database.
    This is a read-only model to access vehicle data.
    """
    __tablename__ = "vehicles"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID)
    ownerId = Column(String, ForeignKey("users.id"))
    vin = Column(String, unique=True)
    make = Column(String, nullable=False)
    model = Column(String, nullable=False)
    year = Column(Integer, nullable=False)
    engineType = Column(String, nullable=False)
    fuelType = Column(String, nullable=False)
    transmissionType = Column(String, nullable=False)
    drivetrain = Column(String, nullable=False)
    licensePlate = Column(String, nullable=False)
    odometerUpdatedAt = Column(DateTime)
    deletedAt = Column(DateTime)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)
    
    # Define relationships
    owner = relationship("User", back_populates="vehicles")
    diagnostics = relationship("Diagnostic", back_populates="vehicle")
    sensorSnapshots = relationship("SensorSnapshot", back_populates="vehicle")
    maintenanceLogs = relationship("MaintenanceLog", back_populates="vehicle") 