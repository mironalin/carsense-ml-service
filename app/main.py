from fastapi import FastAPI

from app.core.config import settings

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="CarSense ML Service API for vehicle diagnostics and predictive maintenance",
    version="0.1.0",
)

@app.get("/")
async def root():
    """Root endpoint with basic service information."""
    return {
        "message": "Welcome to CarSense ML Service API",
        "version": "0.1.0",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)