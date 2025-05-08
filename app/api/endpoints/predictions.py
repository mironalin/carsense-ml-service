from typing import List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from app.core.security import get_current_user, TokenData
from app.schemas.prediction import PredictionRequest, PredictionFeedbackCreate

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

@router.post("/vehicle-health", status_code=status.HTTP_200_OK)
def predict_vehicle_health(
    prediction_input: PredictionRequest = Body(...),
    current_user: TokenData = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Make a prediction about vehicle health based on sensor data and DTCs.

    This endpoint is specifically designed for integration with the CarSense backend.
    It evaluates overall vehicle health, component failure risks, and recommends maintenance.

    Requires authentication. The schema for the request body is PredictionRequest.
    """
    # Access data using Pydantic model attributes
    # vehicle_id is not directly in PredictionRequest, but vehicleInfo.vin or another unique ID could be used
    # For now, let's assume we might use vehicleInfo for context if needed.
    # The existing mock logic doesn't directly use vehicle_id from the top level.

    # sensor_data is now prediction_input.obdParameters or prediction_input.sensorReadings
    # dtc_codes is now prediction_input.dtcCodes

    # Example:
    # vehicle_make = prediction_input.vehicleInfo.make
    # dtc_list = prediction_input.dtcCodes
    # obd_params = prediction_input.obdParameters
    
    # The current mock logic uses hardcoded component failure probabilities
    # and doesn't directly use the input data for calculation.
    # If it were to use it, you'd access it via prediction_input.
    # For example, if you had a model that took dtc_codes:
    # risk_based_on_dtcs = some_model_function(prediction_input.dtcCodes)

    # Mock prediction logic (remains the same for now)
    component_failure_probabilities = {
        "engine": 0.15,
        "transmission": 0.05,
        "battery": 0.30,
        "brakes": 0.10,
        "suspension": 0.20
    }

    # Calculate overall health score (inverse of the highest component risk)
    max_risk = max(component_failure_probabilities.values())
    health_score = max(0, min(100, 100 - max_risk * 100))

    # Generate maintenance recommendations based on component risks
    recommendations = []
    for component, probability in component_failure_probabilities.items():
        if probability > 0.25:
            urgency = "critical"
        elif probability > 0.15:
            urgency = "high"
        elif probability > 0.10:
            urgency = "medium"
        else:
            continue  # Skip low-risk components

        recommendations.append({
            "component": component,
            "urgency": urgency,
            "estimatedCost": {
                "min": 150,
                "max": 500,
                "currency": "RON"
            }
        })

    # Prepare the response
    prediction_result = {
        "prediction": {
            "componentFailureProbability": component_failure_probabilities,
            "vehicleHealthScore": health_score,
            "maintenanceRecommendations": recommendations
        },
        "confidence": 0.85
    }

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