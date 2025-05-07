from typing import List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.ml import MLModel, Prediction
from app.schemas.prediction_db import PredictionCreate, PredictionResponse, PredictionList

router = APIRouter()

@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED)
def create_prediction(
    prediction_data: PredictionCreate,
    db: Session = Depends(get_db)
) -> PredictionResponse:
    """
    Make a new prediction using the specified ML model.
    
    This endpoint receives vehicle data, passes it to the appropriate model,
    and returns prediction results.
    """
    # Check if model exists
    model = db.query(MLModel).filter(MLModel.id == prediction_data.model_id).first()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {prediction_data.model_id} not found"
        )
    
    # Create prediction record
    prediction = Prediction(
        model_id=prediction_data.model_id,
        vehicle_id=prediction_data.vehicle_id,
        prediction_date=datetime.utcnow(),
        input_data=prediction_data.input_data,
        results=prediction_data.results,
        confidence=prediction_data.confidence
    )
    
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    
    return prediction

@router.get("/vehicle/{vehicle_id}", response_model=List[PredictionList])
def get_vehicle_predictions(
    vehicle_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
) -> List[PredictionList]:
    """
    Get prediction history for a specific vehicle.
    """
    predictions = db.query(Prediction)\
        .filter(Prediction.vehicle_id == vehicle_id)\
        .order_by(Prediction.prediction_date.desc())\
        .offset(skip)\
        .limit(limit)\
        .all()
    
    return predictions

@router.get("/{prediction_id}", response_model=PredictionResponse)
def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
) -> PredictionResponse:
    """
    Get detailed information about a specific prediction.
    """
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with id {prediction_id} not found"
        )
    
    return prediction

@router.post("/{prediction_id}/feedback", response_model=PredictionResponse)
def add_feedback(
    prediction_id: int,
    feedback: dict,
    db: Session = Depends(get_db)
) -> PredictionResponse:
    """
    Add feedback to a prediction (e.g., whether it was accurate).
    
    This feedback can be used to improve model performance over time.
    """
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with id {prediction_id} not found"
        )
    
    # Update the prediction with feedback
    prediction.feedback = feedback
    db.commit()
    db.refresh(prediction)
    
    return prediction 