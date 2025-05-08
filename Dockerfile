FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY setup_db.py ./
COPY docker-entrypoint.sh ./

# Make the entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Create a directory for models
RUN mkdir -p models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl --fail http://localhost:8000/api/v1/health/ || exit 1

# Expose the application port
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 