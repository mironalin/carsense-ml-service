"""
Setup script for initializing the ML service database tables.
This only creates the ML-specific tables and does not touch existing tables.
"""

import logging
from app.db.init_db import init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Initialize database tables for the ML service."""
    logger.info("Creating ML service database tables...")
    try:
        init_db()
        logger.info("Database tables created successfully!")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

if __name__ == "__main__":
    setup_database() 