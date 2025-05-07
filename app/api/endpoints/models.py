from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.ml import MLModel
from app.schemas.ml_model import MLModelCreate, MLModelResponse

router = APIRouter()

@router.post("/", response_model=MLModelResponse, status_code=status.HTTP_201_CREATED)
def create_model(
    model_data: MLModelCreate,
    db: Session = Depends(get_db)
) -> MLModelResponse:
    """
    Register a new ML model in the system.

    This endpoint is used when a new model has been trained and needs to be
    registered in the system for use in predictions.
    """
    # Create a new ML model record
    db_model = MLModel(
        name=model_data.name,
        version=model_data.version,
        description=model_data.description,
        trained_at=model_data.trained_at,
        framework=model_data.framework,
        vehicle_make=model_data.vehicle_make,
        model_type=model_data.model_type,
        model_path=model_data.model_path,
        metrics=model_data.metrics,
        input_features=model_data.input_features,
        hyperparameters=model_data.hyperparameters
    )

    db.add(db_model)
    db.commit()
    db.refresh(db_model)

    return db_model

@router.get("/", response_model=List[MLModelResponse])
def list_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[str] = None,
    vehicle_make: Optional[str] = None,
    db: Session = Depends(get_db)
) -> List[MLModelResponse]:
    """
    List all registered ML models, with optional filtering.

    Supports filtering by model_type and vehicle_make.
    """
    query = db.query(MLModel)

    # Apply filters if provided
    if model_type:
        query = query.filter(MLModel.model_type == model_type)
    if vehicle_make:
        query = query.filter(MLModel.vehicle_make == vehicle_make)

    models = query.offset(skip).limit(limit).all()
    return models

@router.get("/{model_id}", response_model=MLModelResponse)
def get_model(
    model_id: int,
    db: Session = Depends(get_db)
) -> MLModelResponse:
    """
    Get detailed information about a specific ML model.
    """
    model = db.query(MLModel).filter(MLModel.id == model_id).first()

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {model_id} not found"
        )

    return model