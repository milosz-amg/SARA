# SARA Database Setup Guide

This guide explains how to set up the SARA PostgreSQL database with all data included.

## Quick Start

### Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- At least 20 GB free disk space
- Internet connection (for initial setup)

### Option 1: With Database Dump File (Recommended)

1. **Download the database dump file** (`devdb_backup.dump`) - approximately 13 GB
2. **Place it in the same directory** as `setup_sara_database.sh`
3. **Run the setup script:**
   ```bash
   chmod +x setup_sara_database.sh
   ./setup_sara_database.sh
   ```

The script will automatically:
- Pull PostgreSQL with pgvector extension
- Create and start a Docker container
- Set up the database and user
- Restore all data from the dump file

### Option 2: Download from URL

If you host the dump file online (GitHub Releases, Google Drive, etc.):

1. **Edit `setup_sara_database.sh`** and update the `DUMP_URL` variable:
   ```bash
   DUMP_URL="https://your-url-here/devdb_backup.dump"
   ```

2. **Run the setup script:**
   ```bash
   chmod +x setup_sara_database.sh
   ./setup_sara_database.sh
   ```

The script will download the dump file automatically.

## Connection Details

After setup completes, connect using:

- **Host:** `localhost`
- **Port:** `5432`
- **Database:** `devdb`
- **User:** `devuser`
- **Password:** `devpass`

### Connect via psql:
```bash
PGPASSWORD=devpass psql -h localhost -U devuser -d devdb -p 5432
```

### Connect via Docker:
```bash
docker exec -it sara-postgres psql -U devuser -d devdb
```

### Connect via Python:
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="devdb",
    user="devuser",
    password="devpass"
)
```

## Database Features

This database includes:
- ✅ **pgvector extension** - for vector similarity search
- ✅ **All UAM data** - authors, works, projects
- ✅ **NCN data** - research projects and proposals
- ✅ **OpenAlex data** - scientific publications

## Managing the Container

### Stop the container:
```bash
docker stop sara-postgres
```

### Start the container:
```bash
docker start sara-postgres
```

### View logs:
```bash
docker logs sara-postgres
```

### Remove the container (keeps data):
```bash
docker rm -f sara-postgres
```

### Completely remove everything:
```bash
docker rm -f sara-postgres
docker volume prune
```

## Troubleshooting

### Port 5432 already in use
If you already have PostgreSQL running on port 5432, you can change the port in the setup script:
```bash
HOST_PORT="5433"  # Use a different port
```

Then connect using the new port: `-p 5433`

### Container already exists
Run:
```bash
docker rm -f sara-postgres
./setup_sara_database.sh
```

### Insufficient disk space
The database requires:
- ~13 GB for the dump file
- ~15 GB for the restored database
- Total: ~30 GB free space recommended

### Restore takes too long
The restore process can take 10-30 minutes depending on your system. Be patient!

## File Structure

```
SARA/
├── setup_sara_database.sh          # Main setup script
├── DATABASE_SETUP_README.md        # This file
└── devdb_backup.dump               # Database dump (13 GB, not in git)
```

## Hosting the Dump File

The dump file is too large for Git (13 GB). Host it using:

### Option A: GitHub Releases (Recommended)
1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Upload `devdb_backup.dump` as a release asset
4. Copy the download URL and update `DUMP_URL` in the script

### Option B: Google Drive
1. Upload `devdb_backup.dump` to Google Drive
2. Right-click → "Get link" → "Anyone with the link"
3. Use this URL format:
   ```
   https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
   ```

### Option C: Dropbox
1. Upload to Dropbox
2. Get the sharing link
3. Change `?dl=0` to `?dl=1` at the end

### Option D: Cloud Storage (AWS S3, Azure Blob, etc.)
Upload to your cloud provider and use the public URL.

## Support

For issues or questions, please open an issue on GitHub.

---

**Note:** Remember to change the default password (`devpass`) in production environments!
