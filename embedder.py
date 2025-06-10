import faiss # Biblioteka do wyszukiwania wektorów (tworzenie indeksu) https://github.com/facebookresearch/faiss
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#funckcja wysyła tekst do modelu OpenAI i zwraca embedding (wektor - listę liczb) dla tego tekstu
def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small", #szybki, tani, skuteczny
        input=text
    )
    return response.data[0].embedding

#wczytujemy dane z researchers.json
def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(data_path: str, index_path: str):
    data = load_data(data_path) #lista naukowców z jsona
    texts = [] #tu bedzie tekst do embeddingu
    metadata = [] #będzie tu trzymana metadane o naukowcach

    #tu łączymy dane (json) w jeden tekst, który będzie użyty do tworzenia embeddingów (bo embeddingi działą na tekstach)
    for researcher in data:
        text = f"{researcher['name']} z {researcher['affiliation']} prowadzi badania nad {', '.join(researcher['research_areas'])}.\n"
        for project in researcher.get("projects", []):
            text += f"Projekt: {project['title']} ({project['years']}, {project['grant_amount']} PLN)\n"
        texts.append(text)
        metadata.append(researcher)

    embeddings = [embed_text(text) for text in texts] #dla każdego tekstu tworzymy embedding
    dimension = len(embeddings[0]) #liczba wymiarów embeddingu (długość wektora)
    index = faiss.IndexFlatL2(dimension) #indeks bazujący na L2 (odległość euklidesowa)
    index.add(np.array(embeddings).astype("float32")) #dodajemy embeddingi do indeksu

    faiss.write_index(index, index_path)
    with open(index_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
