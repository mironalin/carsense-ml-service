"""
SQLAlchemy models for sensor-related tables.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base

class SensorSnapshot(Base):
    """
    Reference to the sensorSnapshots table in the main database.
    """
    __tablename__ = "sensorSnapshots"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID)
    vehicleId = Column(Integer, ForeignKey("vehicles.id"))
    timestamp = Column(DateTime)
    location = Column(JSON)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)
    
    # Define relationships
    vehicle = relationship("Vehicle", back_populates="sensorSnapshots")
    sensorReadings = relationship("SensorReading", back_populates="sensorSnapshot")


class SensorReading(Base):
    """
    Reference to the sensorReadings table in the main database.
    This is a read-only model to access OBD data.
    """
    __tablename__ = "sensorReadings"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID)
    sensorSnapshotsId = Column(Integer, ForeignKey("sensorSnapshots.id"))
    pid = Column(String, nullable=False)  # e.g., "rpm", "temp", "fuel", "speed"
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)  # e.g., "RPM", "°C", "%", "km/h" 
    timestamp = Column(DateTime)
    
    # Define relationship
    sensorSnapshot = relationship("SensorSnapshot", back_populates="sensorReadings") 