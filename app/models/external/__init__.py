"""
This package contains SQLAlchemy models that mirror the structure of the existing
tables in the CarSense backend database. These are used for read-only access
to reference the data without modifying the original schema.
"""

# Import all models to make them available when importing the package
from app.models.external.user import User
from app.models.external.vehicle import Vehicle
from app.models.external.sensor import SensorSnapshot, SensorReading
from app.models.external.diagnostic import Diagnostic, DiagnosticDTC, DTCLibrary
from app.models.external.maintenance import MaintenanceLog, ServiceWorkshop

# Export all models
__all__ = [
    "User",
    "Vehicle",
    "SensorSnapshot",
    "SensorReading",
    "Diagnostic", 
    "DiagnosticDTC",
    "DTCLibrary",
    "MaintenanceLog",
    "ServiceWorkshop"
] 