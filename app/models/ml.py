from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.db.session import Base
from app.db.base_model import BaseModel

class MLModel(Base, BaseModel):
    """
    Store information about trained ML models.
    This table tracks model metadata, versions, and performance metrics.
    """
    __tablename__ = "ml_models"
    
    name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    description = Column(String, nullable=True)
    trained_at = Column(DateTime, nullable=False)
    framework = Column(String, nullable=False)  # e.g., 'tensorflow', 'pytorch', 'sklearn'
    vehicle_make = Column(String, nullable=True)  # For make-specific models
    model_type = Column(String, nullable=False)  # e.g., 'classification', 'regression'
    model_path = Column(String, nullable=False)  # Path to the saved model
    
    # Model performance metrics
    metrics = Column(JSON, nullable=True)  # e.g., {'accuracy': 0.92, 'f1': 0.89}
    input_features = Column(JSON, nullable=False)  # List of features the model expects
    hyperparameters = Column(JSON, nullable=True)  # Hyperparameters used
    
    # Define relationship to predictions
    predictions = relationship("Prediction", back_populates="model")
    
    def __repr__(self):
        return f"<MLModel {self.name} v{self.version}>"


class Prediction(Base, BaseModel):
    """
    Store prediction records for tracking and analysis.
    This table tracks all predictions made by the ML models.
    """
    __tablename__ = "ml_predictions"
    
    # Foreign keys
    model_id = Column(Integer, ForeignKey("ml_models.id"), nullable=False)
    vehicle_id = Column(Integer, nullable=False)  # Reference without constraint
    
    # Prediction details
    prediction_date = Column(DateTime, nullable=False)
    input_data = Column(JSON, nullable=False)  # Input parameters used for prediction
    results = Column(JSON, nullable=False)  # Prediction results
    confidence = Column(Float, nullable=False)  # Confidence score
    feedback = Column(JSON, nullable=True)  # User feedback or actual outcomes
    
    # Define relationship to model
    model = relationship("MLModel", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction {self.id} for vehicle {self.vehicle_id}>" 