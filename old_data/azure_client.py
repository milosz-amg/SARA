# azure_client.py

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Wczytaj zmienne środowiskowe
load_dotenv()

# Klient Azure OpenAI
client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
)

# ⬇⬇⬇ Funkcja, która MUSI być zaimportowana w main.py ⬇⬇⬇
def ask_azure_openai(prompt: str, model: str = "gpt-4o", temperature: float = 1.0) -> str:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=4096,
        temperature=temperature,
        top_p=1.0,
        model=model
    )
    return response.choices[0].message.content
