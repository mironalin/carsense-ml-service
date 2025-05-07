# CarSense ML Service

Machine learning service for the CarSense vehicle diagnostics ecosystem. This service provides predictive maintenance recommendations based on OBD-II data and diagnostic trouble codes.

## Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```
   source .venv/bin/activate  # On Linux/Mac
   .venv\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up database connection:
   
   Create a `.env` file at the root of the project with your database connection settings:
   
   ```
   # Direct connection to Neon PostgreSQL (recommended)
   DATABASE_URL=postgresql://username:password@dbhost:5432/carsense
   ```
   
   This ML service connects to the same database used by the main CarSense backend to access vehicle data, OBD readings, and diagnostic codes.

## Running the Service

Run the development server with:
```
python -m uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API documentation will be available at http://localhost:8000/docs

## Integration with CarSense Backend

This ML service integrates with the existing CarSense backend by:

1. Connecting to the same Neon PostgreSQL database
2. Reading data from the existing tables (vehicles, obd_readings, dtc, etc.)
3. Creating its own tables for ML model metadata and prediction tracking
4. Providing predictions through a REST API that the main backend can call

The service follows the architecture outlined in the CarSense backend documentation. 