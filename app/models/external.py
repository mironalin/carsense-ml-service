"""
This module defines SQLAlchemy models that mirror the structure of the existing
tables in the CarSense backend database. These are used for read-only access
to reference the data without modifying the original schema.

These models are updated based on the actual schema in db-schemas-ts folder.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text, Boolean, ARRAY
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.db.session import Base

class User(Base):
    """
    Reference to the users table in the main database.
    """
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    # Define relationships
    vehicles = relationship("Vehicle", back_populates="owner")


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