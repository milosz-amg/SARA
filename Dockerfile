# Użyj oficjalnego obrazu Pythona
FROM python:3.10-slim

# Ustaw katalog roboczy
WORKDIR /app

# Skopiuj pliki projektu do kontenera
COPY . .

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

# Ustaw port, na którym będzie działać API
EXPOSE 8000

# Komenda uruchamiająca aplikację
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
