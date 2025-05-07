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

## Running the Service

Run the development server with:
```
python -m uvicorn app.main:app --reload
```

The API will be available at http://localhost:8000

API documentation will be available at http://localhost:8000/docs 