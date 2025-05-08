#!/bin/bash
set -e

# Function to check if Postgres is ready
postgres_ready() {
  python << END
import sys
import psycopg2
import os
import time
import re
from urllib.parse import urlparse

# Get database URL from environment
db_url = os.environ.get("DATABASE_URL")
if not db_url:
    print("DATABASE_URL not set")
    sys.exit(1)

print(f"Attempting to connect with DATABASE_URL (credentials hidden)")

# Use urlparse to parse the connection string safely
try:
    # Parse the URL
    result = urlparse(db_url)

    # Extract components
    username = result.username
    password = result.password
    hostname = result.hostname
    port = result.port
    path = result.path

    # Remove leading slash from path to get database name
    dbname = path.lstrip('/')

    # Try to connect
    conn = psycopg2.connect(
        dbname=dbname,
        user=username,
        password=password,
        host=hostname,
        port=port,
        connect_timeout=5  # Add a timeout for connection attempts
    )
    conn.close()
    print("Successfully connected to PostgreSQL database")
    sys.exit(0)
except Exception as e:
    print(f"Can't connect to PostgreSQL: {e}")
    sys.exit(1)
END
}

# Wait for Postgres to be ready with timeout
echo "Checking connection to PostgreSQL (Neon)..."
attempt=1
max_attempts=30  # Maximum number of attempts (30 seconds)

until postgres_ready; do
  echo "PostgreSQL connection attempt ${attempt}/${max_attempts}"

  if [ $attempt -ge $max_attempts ]; then
    echo "Could not connect to PostgreSQL after ${max_attempts} attempts - exiting"
    exit 1
  fi

  attempt=$((attempt+1))
  sleep 1
done
echo "PostgreSQL is accessible - continuing"

# Initialize the database
echo "Initializing the database..."
python setup_db.py

# Start the application
echo "Starting the ML service..."
exec "$@"