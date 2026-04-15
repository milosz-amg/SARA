# Quick Start - For Your Friends

## Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- 20 GB free disk space

## Setup (2 Steps)

### Step 1: Get the Files
```bash
git clone https://github.com/YOUR_USERNAME/SARA.git
cd SARA
```

### Step 2: Run Setup
```bash
chmod +x setup_sara_database.sh
./setup_sara_database.sh
```

That's it! The script will:
- Download PostgreSQL with pgvector
- Create a Docker container
- Download the database (13 GB)
- Restore all data automatically

⏱️ Setup takes 15-30 minutes depending on your internet and hardware.

## Connect to Database

After setup completes:

```bash
# Using psql
PGPASSWORD=devpass psql -h localhost -U devuser -d devdb

# Or using Docker
docker exec -it sara-postgres psql -U devuser -d devdb
```

### Connection Details
- Host: `localhost`
- Port: `5432`
- Database: `devdb`
- User: `devuser`
- Password: `devpass`

## Test It

```sql
-- Count authors
SELECT COUNT(*) FROM authors;

-- Test vector similarity search
SELECT display_name,
       embedding <-> (SELECT embedding FROM authors LIMIT 1) as distance
FROM authors
WHERE embedding IS NOT NULL
ORDER BY distance
LIMIT 5;
```

## Need Help?

See [DATABASE_SETUP_README.md](DATABASE_SETUP_README.md) for detailed documentation.
