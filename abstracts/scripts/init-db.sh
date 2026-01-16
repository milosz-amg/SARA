#!/bin/bash
set -e

# This script runs automatically when the container first starts
# It sets up the pgvector extension and creates the devuser role

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create the vector extension
    CREATE EXTENSION IF NOT EXISTS vector;
    
    -- Create devuser role (from the original database)
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'devuser') THEN
            CREATE ROLE devuser LOGIN PASSWORD '1q2w3e';
        END IF;
    END
    \$\$;
    
    -- Grant necessary privileges
    GRANT ALL PRIVILEGES ON DATABASE $POSTGRES_DB TO devuser;
    GRANT ALL PRIVILEGES ON SCHEMA public TO devuser;
    
    -- Show success
    SELECT 'Database initialization complete' as status;
EOSQL