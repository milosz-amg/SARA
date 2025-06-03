# SARA
Seek and research assistant


## Uruchomienie
1. Odpalić VM na Azure
2. Połączyć się z SSH: `ssh -i ~/.ssh/sara_key.pem milosz@4.210.218.170`
3. Dodać OPEN_AI_APIKEY do .env
4. Uruchomić docker na VM: `docker compose up --build`
5. w przeglądarce: `http://sara.westeurope.cloudapp.azure.com:8000/docs`