"""
Import all models from their respective modules.
"""

# Import ML models
from app.models.ml import MLModel, Prediction

# Import external models
from app.models.external import (
    User,
    Vehicle,
    SensorSnapshot,
    SensorReading,
    Diagnostic,
    DiagnosticDTC,
    DTCLibrary,
    MaintenanceLog,
    ServiceWorkshop
)

# Export all models
__all__ = [
    # ML models
    "MLModel",
    "Prediction",
    
    # External reference models
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