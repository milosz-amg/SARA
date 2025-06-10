import os
from embedder import build_faiss_index
from search import search_faiss
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Ścieżki
DATA_PATH = "data/researchers/researchers.json"
INDEX_PATH = "faiss_index/uam.index"

# Upewnij się, że katalog na indeks istnieje
os.makedirs("faiss_index", exist_ok=True)

# Tworzenie indeksu FAISS
print("🔧 Tworzę indeks FAISS...")
build_faiss_index(DATA_PATH, INDEX_PATH)
print("✅ Indeks gotowy!")

# Inicjalizacja klienta OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Funkcja główna: pytanie + wyszukiwanie + odpowiedź
def ask_with_context(query):
    print(f"🔎 Szukam kontekstu dla pytania: {query}")
    results = search_faiss(query, INDEX_PATH, top_k=3)

    context = ""
    for res in results:
        context += f"{res['name']} ({res['affiliation']}): {', '.join(res['research_areas'])}\n"
        for project in res.get("projects", []):
            context += f"- {project['title']} ({project['years']}) | {project['grant_amount']} PLN\n"
        context += f"Źródło: {res['source']}\n\n"

    prompt = f"""
KONTEKST:
{context}

PYTANIE:
{query}

ODPOWIEDŹ:
"""

    print("📡 Wysyłam do modelu GPT...")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    answer = response.choices[0].message.content
    return answer

def print_results(results):
    for res in results:
        print(f"\n🔹 {res['name']} ({res['affiliation']})")
        print(f"   Dziedziny: {', '.join(res['research_areas'])}")
        for project in res.get("projects", []):
            print(f"   📁 Projekt: {project['title']} ({project['years']}) – {project['grant_amount']} PLN")
        print(f"   Źródło: {res['source']}")

if __name__ == "__main__":
    question = "Którzy naukowcy z WMiI UAM zajmują się logiką rozmytą?"
    results = search_faiss(question, INDEX_PATH, top_k=3)
    print_results(results)
