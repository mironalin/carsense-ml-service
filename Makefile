.PHONY: help build up down logs ps shell test clean

# Display help information
help:
	@echo "CarSense ML Service"
	@echo "================================================================================"
	@echo "make build        - Build the Docker images"
	@echo "make up           - Start the containers"
	@echo "make down         - Stop the containers"
	@echo "make logs         - Show logs from all containers"
	@echo "make ps           - Show running containers"
	@echo "make shell        - Start a shell in the ml-service container"
	@echo "make test         - Run tests"
	@echo "make clean        - Remove all containers, volumes, and images"
	@echo "================================================================================"

# Build Docker images
build:
	docker compose build

# Start containers
up:
	docker compose up -d

# Start containers and show logs
up-logs:
	docker compose up

# Stop containers
down:
	docker compose down

# Show logs
logs:
	docker compose logs -f

# Show running containers
ps:
	docker compose ps

# Start a shell in the ml-service container
shell:
	docker compose exec ml-service bash

# Run tests
test:
	docker compose exec ml-service pytest

# Clean everything
clean:
	docker compose down -v --rmi all 