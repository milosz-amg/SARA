#!/usr/bin/env bash
set -e

#############################################################################
# setup_sara_database.sh
#
# Complete database setup script for SARA project
# This script will:
# 1. Pull PostgreSQL with pgvector extension
# 2. Create and start the container
# 3. Download the database dump from your storage
# 4. Restore all data automatically
#
# Usage: ./setup_sara_database.sh
#############################################################################

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
POSTGRES_IMAGE="pgvector/pgvector:pg15"
CONTAINER_NAME="sara-postgres"
DB_NAME="devdb"
DB_USER="devuser"
DB_PASS="devpass"
HOST_PORT="5432"
MAX_WAIT_SECONDS=60

# URL where you'll host the dump file (update this!)
# Options: Google Drive, Dropbox, GitHub Releases, your own server, etc.
DUMP_URL="YOUR_DUMP_URL_HERE"
# Example: "https://github.com/youruser/yourrepo/releases/download/v1.0/devdb_backup.dump"
# Example: "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SARA Database Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[ERROR] Docker is not installed. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if dump URL is configured
if [ "$DUMP_URL" = "YOUR_DUMP_URL_HERE" ]; then
    echo -e "${YELLOW}[WARNING] Dump URL not configured!${NC}"
    echo "Please edit this script and set DUMP_URL to the location of your database dump."
    echo ""
    echo "OR run this script with the dump file in the same directory:"
    echo "  Place 'devdb_backup.dump' in the same folder as this script"
    echo ""

    if [ ! -f "devdb_backup.dump" ]; then
        echo -e "${RED}[ERROR] devdb_backup.dump not found in current directory.${NC}"
        exit 1
    else
        echo -e "${GREEN}[INFO] Found devdb_backup.dump in current directory!${NC}"
        DUMP_FILE="./devdb_backup.dump"
    fi
else
    echo -e "${GREEN}[INFO] Will download database dump from: $DUMP_URL${NC}"
fi

echo ""
echo -e "${GREEN}[INFO] Configuration:${NC}"
echo "  Image:          $POSTGRES_IMAGE"
echo "  Container:      $CONTAINER_NAME"
echo "  Database:       $DB_NAME"
echo "  User:           $DB_USER"
echo "  Port:           $HOST_PORT"
echo ""

# Check if container already exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}[WARNING] Container '$CONTAINER_NAME' already exists!${NC}"
    read -p "Do you want to remove it and start fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}[INFO] Removing existing container...${NC}"
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    else
        echo -e "${RED}[ERROR] Cannot proceed with existing container. Exiting.${NC}"
        exit 1
    fi
fi

# Pull PostgreSQL image with pgvector
echo -e "${BLUE}[INFO] Pulling PostgreSQL image with pgvector extension...${NC}"
docker pull "$POSTGRES_IMAGE"

# Create and start container
echo -e "${BLUE}[INFO] Creating and starting PostgreSQL container...${NC}"
docker run -d \
    --name "$CONTAINER_NAME" \
    -e POSTGRES_PASSWORD="$DB_PASS" \
    -p "${HOST_PORT}:5432" \
    "$POSTGRES_IMAGE"

# Wait for PostgreSQL to be ready
echo -e "${BLUE}[INFO] Waiting for PostgreSQL to be ready...${NC}"
ELAPSED=0
until docker exec "$CONTAINER_NAME" pg_isready -U postgres >/dev/null 2>&1; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [ $ELAPSED -ge $MAX_WAIT_SECONDS ]; then
        echo -e "${RED}[ERROR] Timeout waiting for PostgreSQL to be ready${NC}"
        exit 1
    fi
    echo -n "."
done
echo ""
echo -e "${GREEN}[INFO] PostgreSQL is ready!${NC}"

# Create user and database
echo -e "${BLUE}[INFO] Creating database user and database...${NC}"
docker exec "$CONTAINER_NAME" psql -U postgres -c \
    "CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASS';" 2>/dev/null || echo "User already exists"

docker exec "$CONTAINER_NAME" psql -U postgres -c \
    "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null || echo "Database already exists"

# Enable pgvector extension
echo -e "${BLUE}[INFO] Enabling pgvector extension...${NC}"
docker exec "$CONTAINER_NAME" psql -U postgres -d "$DB_NAME" -c \
    "CREATE EXTENSION IF NOT EXISTS vector;"

# Download or use local dump file
if [ -z "$DUMP_FILE" ]; then
    echo -e "${BLUE}[INFO] Downloading database dump...${NC}"
    DUMP_FILE="./devdb_backup.dump"

    # Download with progress
    if command -v wget &> /dev/null; then
        wget -O "$DUMP_FILE" "$DUMP_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$DUMP_FILE" "$DUMP_URL"
    else
        echo -e "${RED}[ERROR] Neither wget nor curl found. Cannot download dump file.${NC}"
        exit 1
    fi
fi

# Copy dump file to container
echo -e "${BLUE}[INFO] Copying dump file to container...${NC}"
docker cp "$DUMP_FILE" "$CONTAINER_NAME:/tmp/devdb_backup.dump"

# Restore database
echo -e "${BLUE}[INFO] Restoring database (this may take several minutes)...${NC}"
docker exec "$CONTAINER_NAME" pg_restore \
    -U postgres \
    -d "$DB_NAME" \
    --verbose \
    /tmp/devdb_backup.dump 2>&1 | grep -E "(processing|creating|restoring)" || true

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Database setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Connection details:${NC}"
echo "  Host:     localhost"
echo "  Port:     $HOST_PORT"
echo "  Database: $DB_NAME"
echo "  User:     $DB_USER"
echo "  Password: $DB_PASS"
echo ""
echo -e "${GREEN}Quick connection commands:${NC}"
echo "  psql:     PGPASSWORD=$DB_PASS psql -h localhost -U $DB_USER -d $DB_NAME -p $HOST_PORT"
echo "  Docker:   docker exec -it $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME"
echo ""
echo -e "${GREEN}Container management:${NC}"
echo "  Stop:     docker stop $CONTAINER_NAME"
echo "  Start:    docker start $CONTAINER_NAME"
echo "  Remove:   docker rm -f $CONTAINER_NAME"
echo ""
echo -e "${YELLOW}Note: The database dump file is stored in the container at /tmp/devdb_backup.dump${NC}"
echo ""
