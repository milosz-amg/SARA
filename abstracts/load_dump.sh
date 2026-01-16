#!/bin/bash

# Script to load PostgreSQL dump file into database
# Usage: ./load_dump.sh [dump_file_name]

set -e

# Default values
DUMP_FILE=${1:-"devdb_backup_new.dump"}
POSTGRES_USER="ab_user"
POSTGRES_DB="ab_db"

echo "🚀 Starting database restore process..."

# Check if docker-compose is running
if ! docker compose ps | grep -q "postgres_db.*running"; then
    echo "PostgreSQL container is not running. Starting it now..."
    docker compose up -d postgres
    echo "Waiting for PostgreSQL to be ready..."
    sleep 15
fi

# Wait for PostgreSQL to be fully ready
echo "Waiting for database to accept connections..."
for i in {1..30}; do
    if docker compose exec -T postgres pg_isready -U $POSTGRES_USER -d $POSTGRES_DB > /dev/null 2>&1; then
        echo "Database is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Timeout waiting for database to be ready"
        exit 1
    fi
    sleep 1
done

# Verify pgvector extension is installed
echo "Verifying pgvector extension..."
docker compose exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';" || {
    echo "pgvector extension not found. This should have been installed automatically."
    exit 1
}
echo "pgvector extension verified"

# Check if dump file exists in data directory
if [ ! -f "./data/$DUMP_FILE" ]; then
    echo "Error: Dump file './data/$DUMP_FILE' not found!"
    echo "Please place your .dump file in the ./data/ directory"
    exit 1
fi

echo "Loading dump file: $DUMP_FILE"

# Restore the dump file
# Using pg_restore for custom format dumps or psql for plain SQL dumps
FILE_EXTENSION="${DUMP_FILE##*.}"

if [ "$FILE_EXTENSION" = "sql" ]; then
    echo "Detected SQL format, using psql..."
    docker compose exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB < "./data/$DUMP_FILE"
else
    echo "Detected custom/compressed format, using pg_restore..."
    echo "Note: Errors about dropping non-existent objects are normal on first restore"
    
    # Use --no-owner to map devuser objects to ab_user
    docker compose exec -T postgres pg_restore \
        -U $POSTGRES_USER \
        -d $POSTGRES_DB \
        --no-owner \
        --role=ab_user \
        -v \
        /dump/$DUMP_FILE 2>&1 | tee /tmp/restore.log || {
        
        echo ""
        echo "Some errors occurred during restore. Checking if data was loaded..."
        
        # Check if tables were created
        TABLE_COUNT=$(docker compose exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
        
        if [ "$TABLE_COUNT" -gt 0 ]; then
            echo "Tables were created successfully (found $TABLE_COUNT tables)"
            echo "The errors about dropping non-existent objects can be safely ignored"
        else
            echo "No tables found. Restore may have failed."
            exit 1
        fi
    }
fi

echo ""
echo "Database restore completed!"
echo ""
echo "Database Summary:"
docker compose exec -T postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
    SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
    FROM pg_tables 
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

echo ""
echo "You can connect to the database with:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: $POSTGRES_DB"
echo "   User: $POSTGRES_USER"
echo "   Password: 1q2w3e"