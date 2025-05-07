from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class MLModelBase(BaseModel):
    """Base schema for ML model information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    description: Optional[str] = Field(None, description="Model description")
    framework: str = Field(..., description="ML framework used (e.g., 'tensorflow', 'pytorch', 'sklearn')")
    vehicle_make: Optional[str] = Field(None, description="Vehicle make if model is make-specific")
    model_type: str = Field(..., description="Model type (e.g., 'classification', 'regression')")
    
class MLModelCreate(MLModelBase):
    """Schema for creating a new ML model entry."""
    trained_at: datetime = Field(default_factory=datetime.utcnow, description="Training timestamp")
    model_path: str = Field(..., description="Path to the saved model")
    metrics: Dict[str, float] = Field(default={}, description="Model performance metrics")
    input_features: List[str] = Field(..., description="List of features the model expects")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Hyperparameters used for training")

class MLModelResponse(MLModelBase):
    """Schema for ML model response."""
    id: int = Field(..., description="Model ID")
    trained_at: datetime = Field(..., description="Training timestamp")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    input_features: List[str] = Field(..., description="List of features the model expects")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "example": {
                "id": 1,
                "name": "CarSense-DTC-Predictor",
                "version": "1.0.0",
                "description": "DTC code predictive model for Romanian vehicles",
                "framework": "tensorflow",
                "vehicle_make": "Dacia",
                "model_type": "classification",
                "trained_at": "2023-05-01T10:00:00Z",
                "metrics": {"accuracy": 0.92, "f1": 0.89, "precision": 0.90, "recall": 0.87},
                "input_features": ["coolant_temp", "engine_load", "rpm", "speed", "intake_pressure"],
                "created_at": "2023-05-01T12:00:00Z",
                "updated_at": "2023-05-01T12:00:00Z"
            }
        }
    } 