"""
SQLAlchemy models for maintenance-related tables.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base

class MaintenanceLog(Base):
    """
    Reference to the maintenanceLogs table in the main database.
    This is a read-only model to access maintenance history.
    """
    __tablename__ = "maintenanceLogs"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID)
    vehicleId = Column(Integer, ForeignKey("vehicles.id"))
    serviceDate = Column(DateTime, nullable=False)
    description = Column(Text, nullable=False)
    mileage = Column(Integer)
    serviceType = Column(String)
    cost = Column(Float)
    currency = Column(String)
    serviceWorkshopId = Column(Integer, ForeignKey("serviceWorkshops.id"))
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime)
    
    # Define relationship to vehicle
    vehicle = relationship("Vehicle", back_populates="maintenanceLogs")
    workshop = relationship("ServiceWorkshop")


class ServiceWorkshop(Base):
    """
    Reference to the serviceWorkshops table in the main database.
    """
    __tablename__ = "serviceWorkshops"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID)
    name = Column(String, nullable=False)
    address = Column(String)
    location = Column(JSON)
    createdAt = Column(DateTime)
    updatedAt = Column(DateTime) 