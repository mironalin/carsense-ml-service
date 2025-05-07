from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, computed_field
from datetime import datetime

class PredictionCreate(BaseModel):
    """Schema for creating a prediction record in the database."""
    model_id: int = Field(..., description="ID of the ML model used")
    vehicle_id: int = Field(..., description="ID of the vehicle")
    input_data: Dict[str, Any] = Field(..., description="Input data used for prediction")
    results: Dict[str, Any] = Field(..., description="Prediction results")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")

class PredictionResponse(BaseModel):
    """Schema for returning a prediction record from the database."""
    id: int = Field(..., description="Prediction ID")
    model_id: int = Field(..., description="ID of the ML model used")
    vehicle_id: int = Field(..., description="ID of the vehicle")
    prediction_date: datetime = Field(..., description="Date when prediction was made")
    input_data: Dict[str, Any] = Field(..., description="Input data used for prediction")
    results: Dict[str, Any] = Field(..., description="Prediction results")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    feedback: Optional[Dict[str, Any]] = Field(None, description="User feedback or actual outcomes")
    created_at: datetime = Field(..., description="Record creation timestamp")
    updated_at: datetime = Field(..., description="Record update timestamp")

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "model_id": 2,
                "vehicle_id": 5,
                "prediction_date": "2023-05-15T14:30:00Z",
                "input_data": {
                    "coolant_temp": 95,
                    "engine_load": 45,
                    "rpm": 1500,
                    "speed": 60,
                    "voltage": 14.2
                },
                "results": {
                    "component_health": {
                        "cooling_system": 0.85,
                        "electrical_system": 0.96,
                        "engine": 0.92
                    },
                    "maintenance_recommendations": [
                        {
                            "component": "cooling_system",
                            "action": "Check coolant level",
                            "urgency": "medium"
                        }
                    ]
                },
                "confidence": 0.89,
                "feedback": None,
                "created_at": "2023-05-15T14:30:00Z",
                "updated_at": "2023-05-15T14:30:00Z"
            }
        }
    }

class PredictionList(BaseModel):
    """Schema for listing predictions with less detail."""
    id: int = Field(..., description="Prediction ID")
    vehicle_id: int = Field(..., description="ID of the vehicle")
    prediction_date: datetime = Field(..., description="Date when prediction was made")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    feedback: Optional[Dict[str, Any]] = Field(None, description="User feedback or actual outcomes")

    @computed_field
    @property
    def has_feedback(self) -> bool:
        """Whether feedback has been provided."""
        return self.feedback is not None

    model_config = {
        "from_attributes": True
    }