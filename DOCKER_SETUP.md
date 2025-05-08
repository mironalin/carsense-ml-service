# Docker Setup for CarSense ML Service

This document explains how to use Docker to run the CarSense ML Service for development and testing.

## Prerequisites

- Docker with Docker Compose plugin installed on your system
  - Modern versions of Docker (19.03.0+) include Docker Compose as a plugin
- Git (to clone the repository)
- Neon PostgreSQL account (or other managed PostgreSQL service)

## Quick Start

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/carsense-ml-service.git
   cd carsense-ml-service
   ```

2. Create a `.env` file with your Neon PostgreSQL credentials:
   ```
   # Database Configuration
   DATABASE_URL=postgresql://username:password@hostname:port/database_name

   # ML Model Settings
   MODEL_PATH=models
   MIN_PREDICTION_CONFIDENCE=0.6

   # API Settings
   API_V1_STR=/api/v1
   PROJECT_NAME=CarSense ML Service

   # CORS Settings (comma-separated URLs)
   BACKEND_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
   ```

3. Start the services
   ```bash
   make up
   ```

4. Check logs (in a separate terminal)
   ```bash
   make logs
   ```

5. Access the API documentation
   Open http://localhost:8000/api/v1/docs in your browser

6. Shut down the services when done
   ```bash
   make down
   ```

## Commands (using Makefile)

- `make build` - Build the Docker images
- `make up` - Start the containers in detached mode
- `make up-logs` - Start the containers and show logs
- `make down` - Stop the containers
- `make logs` - Show logs from all containers
- `make ps` - Show running containers
- `make shell` - Start a shell in the ml-service container
- `make test` - Run tests
- `make clean` - Remove all containers, volumes, and images

## Configuration

- Environment variables are set in the `.env` file
- The service uses Neon PostgreSQL (or any other PostgreSQL service) specified by DATABASE_URL

## Docker Compose Services

The Docker Compose setup includes the following service:

1. **ml-service**: The FastAPI application running the ML service
   - Exposes port 8000 for API access
   - Connects to your Neon PostgreSQL database using DATABASE_URL
   - Hot-reloads when code changes are made

## Using Neon PostgreSQL

This service is configured to work with [Neon PostgreSQL](https://neon.tech/), a serverless PostgreSQL service:

1. Create an account on Neon and set up a new project
2. Create a database for the ML service
3. Get your connection string from the Neon dashboard
4. Add the connection string to your `.env` file as `DATABASE_URL`

Benefits of using Neon:
- Serverless architecture with auto-scaling
- No need to manage a local database container
- Accessible from anywhere (development, staging, production)
- Built-in branching and time-travel features

## Development with Docker

For development, the application code is mounted as a volume, so changes made to the code will be immediately reflected in the running container (thanks to the `--reload` flag in uvicorn).

To add a new package to the project:

1. Add it to `requirements.txt`
2. Rebuild and restart the containers:
   ```bash
   make down
   make build
   make up
   ```

## Troubleshooting

- **Database connection issues**:
  - Check if your Neon PostgreSQL is accessible and the connection string is correct
  - Verify that your IP is in the allowed list in Neon's connection settings
  - Check the logs with `make logs` to see specific database connection errors
- **API not accessible**: Make sure port 8000 is not in use by another application
- **Container won't start**: Check the logs with `make logs` to see the specific error