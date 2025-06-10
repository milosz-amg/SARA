# Dockerfile

# Bazowy obraz z Pythonem
FROM python:3.11-slim

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj pliki do kontenera
COPY . .

FROM python:3.11-slim

WORKDIR /app

# 🔧 Zainstaluj wymagane systemowe zależności (dla gcc i NetfilterQueue)
RUN apt-get update && apt-get install -y \
    gcc \
    libnetfilter-queue-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Ustaw port, który będzie wystawiony
EXPOSE 8000

# Komenda do uruchomienia serwera
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
