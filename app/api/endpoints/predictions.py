from typing import List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from app.core.security import get_current_user, TokenData
from app.schemas.prediction import PredictionRequest, PredictionFeedbackCreate
from app.services.vehicle_prediction_service import generate_vehicle_health_prediction

from app.db.session import get_db
from app.models.ml import MLModel, Prediction
from app.schemas.prediction_db import PredictionCreate, PredictionResponse, PredictionList

import logging

logger = logging.getLogger(__name__)

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

@router.post("/vehicle-health", status_code=status.HTTP_200_OK)
def predict_vehicle_health(
    prediction_input: PredictionRequest = Body(...),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Make a prediction about vehicle health based on sensor data and DTCs.

    This endpoint is specifically designed for integration with the CarSense backend.
    It evaluates overall vehicle health, component failure risks, and recommends maintenance.

    Requires authentication. The schema for the request body is PredictionRequest.
    """
    # Authorization check:
    # If the current_user.role is 'admin', it implies a trusted call (e.g., from the backend service)
    # and bypasses the specific user-vehicle ownership check.
    if current_user.role != "admin":
        # This is a normal user, a more granular check is needed.
        # TODO: Implement proper user-vehicle ownership check against the database.
        # This requires querying if a vehicle with prediction_input.vehicle_id
        # is associated with current_user.username (which is current_user.sub from the token).
        # Example (assuming Vehicle model has an owner_id or similar field linked to user ID):
        # vehicle = db.query(Vehicle).filter(
        #     Vehicle.id == prediction_input.vehicle_id, 
        #     Vehicle.owner_id == current_user.username
        # ).first()
        # if not vehicle:
        #     raise HTTPException(
        #         status_code=status.HTTP_403_FORBIDDEN,
        #         detail="User not authorized to make predictions for this vehicle."
        #     )
        
        # For now, log a warning and allow the request to proceed for easier development.
        logger.warning(
            f"User '{current_user.username}' (role: {current_user.role}) accessed vehicle_id {prediction_input.vehicle_id}. "
            f"Ownership check not yet fully implemented. Request allowed for now."
        )

    # If authorization passes (or is allowed for now), proceed:
    prediction_result = generate_vehicle_health_prediction(
        prediction_input=prediction_input,
        db=db
    )
    return prediction_result

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
    feedback_input: PredictionFeedbackCreate,
    db: Session = Depends(get_db)
) -> PredictionResponse:
    """
    Add feedback to a prediction (e.g., whether it was accurate).

    This feedback can be used to improve model performance over time.
    The request body should conform to the PredictionFeedbackCreate schema.
    """
    prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()

    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with id {prediction_id} not found"
        )

    # Update the prediction with feedback, converting Pydantic model to dict
    prediction.feedback = feedback_input.model_dump(exclude_unset=True)
    db.commit()
    db.refresh(prediction)

    return prediction