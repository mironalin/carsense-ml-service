import logging
from sqlalchemy.exc import SQLAlchemyError

from app.db.session import engine
from app.models.ml import MLModel, Prediction

logger = logging.getLogger(__name__)

def init_db():
    """
    Initialize the database by creating ML-specific tables.
    Does not touch existing tables from the main CarSense backend.
    """
    try:
        # Create our ML-specific tables
        tables = [MLModel.__table__, Prediction.__table__]
        for table in tables:
            table.create(engine, checkfirst=True)
            logger.info(f"Table {table.name} created successfully")
        
        logger.info("ML service tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {e}")
        raise