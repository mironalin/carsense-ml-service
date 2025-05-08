from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Vehicle information schema
class VehicleInfo(BaseModel):
    """Schema for vehicle information."""
    make: str = Field(..., description="Vehicle manufacturer (e.g., 'Dacia', 'Volkswagen')")
    model: str = Field(..., description="Vehicle model (e.g., 'Logan', 'Golf')")
    year: int = Field(..., description="Manufacturing year")
    vin: Optional[str] = Field(None, description="Vehicle Identification Number")
    engineType: str = Field(..., description="Engine type (e.g., 'diesel', 'gasoline')")
    fuelType: str = Field(..., description="Fuel type")
    transmissionType: Optional[str] = Field(None, description="Transmission type")
    mileage: Optional[int] = Field(None, description="Current vehicle mileage in kilometers")

# OBD sensor reading schema
class SensorReading(BaseModel):
    """Schema for individual sensor reading."""
    pid: str = Field(..., description="Parameter ID (e.g., 'rpm', 'coolant_temp')")
    value: float = Field(..., description="Sensor value")
    unit: str = Field(..., description="Unit of measurement (e.g., 'RPM', '°C')")

# Prediction request schema
class PredictionRequest(BaseModel):
    """Schema for prediction request."""
    vehicleInfo: VehicleInfo = Field(..., description="Vehicle information")
    dtcCodes: List[str] = Field(default=[], description="List of DTC codes present in the vehicle")
    obdParameters: Dict[str, float] = Field(
        default={}, 
        description="OBD parameters as key-value pairs (e.g., {'rpm': 1200, 'coolant_temp': 90})"
    )
    sensorReadings: Optional[List[SensorReading]] = Field(
        default=None, 
        description="Detailed sensor readings with units"
    )
    requestTime: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")

# Component failure probability
class ComponentFailure(BaseModel):
    """Schema for component failure probability."""
    component: str = Field(..., description="Vehicle component name")
    failureProbability: float = Field(..., description="Probability of failure (0.0 to 1.0)")
    timeToFailure: Optional[int] = Field(None, description="Estimated time to failure in days")
    confidence: float = Field(..., description="Confidence in the prediction (0.0 to 1.0)")
    severity: str = Field(..., description="Severity of failure ('low', 'medium', 'high', 'critical')")

# Maintenance recommendation
class MaintenanceRecommendation(BaseModel):
    """Schema for maintenance recommendation."""
    action: str = Field(..., description="Recommended action")
    urgency: str = Field(..., description="Urgency level ('routine', 'soon', 'urgent', 'immediate')")
    component: str = Field(..., description="Target component")
    estimatedCost: Optional[Dict[str, float]] = Field(None, description="Min/max cost estimate")
    description: str = Field(..., description="Detailed description of the recommendation")

# Prediction results
class PredictionResult(BaseModel):
    """Schema for prediction results."""
    vehicleHealthScore: float = Field(..., description="Overall vehicle health score (0-100)")
    componentFailures: List[ComponentFailure] = Field(..., description="Component failure probabilities")
    maintenanceRecommendations: List[MaintenanceRecommendation] = Field(..., description="Maintenance recommendations")
    overallUrgency: str = Field(..., description="Overall urgency level ('low', 'medium', 'high', 'critical')")

# Prediction response
class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    requestId: str = Field(..., description="Unique request identifier")
    predictions: PredictionResult = Field(..., description="Prediction results")
    modelInfo: Dict[str, Union[str, float]] = Field(..., description="Information about the model used")
    processedAt: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "requestId": "f8c3de3d-1fea-4d7c-a8b0-29f63c4c3454",
                "predictions": {
                    "vehicleHealthScore": 87.5,
                    "componentFailures": [
                        {
                            "component": "Mass Airflow Sensor",
                            "failureProbability": 0.78,
                            "timeToFailure": 45,
                            "confidence": 0.85,
                            "severity": "medium"
                        }
                    ],
                    "maintenanceRecommendations": [
                        {
                            "action": "Replace Mass Airflow Sensor",
                            "urgency": "soon",
                            "component": "Mass Airflow Sensor",
                            "estimatedCost": {"min": 150, "max": 300, "currency": "EUR"},
                            "description": "The Mass Airflow Sensor is showing signs of degradation. Replacement within 45 days is recommended."
                        }
                    ],
                    "overallUrgency": "medium"
                },
                "modelInfo": {
                    "name": "CarSense-MAS-Predictor",
                    "version": "1.2.0",
                    "accuracy": 0.92,
                    "lastTraining": "2023-04-15"
                },
                "processedAt": "2023-05-10T14:23:53.012Z"
            }
        }

# PredictionFeedbackCreate schema
class PredictionFeedbackCreate(BaseModel):
    """Schema for submitting feedback on a prediction."""
    accuracy: float = Field(..., description="Accuracy rating or score (e.g., 0.0-1.0, or a rating like 1.0-5.0). Definition TBD by application logic.")
    comments: Optional[str] = Field(None, description="User comments about the prediction.")
    is_actionable: Optional[bool] = Field(None, alias="isActionable", description="Was the prediction actionable by the user?")
    additional_data: Optional[Dict[str, Any]] = Field(None, alias="additionalData", description="Any other relevant feedback data.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "accuracy": 0.85,
                "comments": "The prediction was spot on, helped identify the issue quickly.",
                "isActionable": True,
                "additionalData": {
                    "service_performed": "Replaced MAF sensor",
                    "cost": 150.75
                }
            }
        },
        "populate_by_name": True
    } 