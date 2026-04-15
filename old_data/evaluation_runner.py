import os
import pandas as pd
import json
from dotenv import load_dotenv
from openai import OpenAI
from old_data.embedder import build_faiss_index
from old_data.search import search_faiss
from utils.openai import call_openai

# === KONFIGURACJA ===
DATA_PATH = "data/researchers/researchers.json"
INDEX_PATH = "faiss_index/uam.index"
EXCEL_PATH = "porownanie_RAG_vs_Web.xlsx"
QUESTIONS_PATH = "questions.txt"
TOP_K = 3

load_dotenv()
os.makedirs("faiss_index", exist_ok=True)

# === UTWÓRZ PLIK EXCEL JEŚLI NIE ISTNIEJE ===
if not os.path.exists(EXCEL_PATH):
    df = pd.DataFrame(columns=[
        "Pytanie",
        "Odpowiedź RAG",
        "Odpowiedź Web",
        "Factual Accuracy (RAG)",
        "Completeness (RAG)",
        "Factual Accuracy (Web)",
        "Completeness (Web)",
        "Uwagi"
    ])
    df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")

# === BUDUJ INDEKS FAISS ===
build_faiss_index(DATA_PATH, INDEX_PATH)

# === INICJALIZACJA KLIENTA OPENAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === FUNKCJA: RAG ===
def answer_from_rag(query, top_k=TOP_K):
    results = search_faiss(query, INDEX_PATH, top_k=top_k)
    context = ""
    for res in results:
        context += f"{res['name']} ({res['affiliation']}): {', '.join(res['research_areas'])}\n"
        for project in res.get("projects", []):
            context += f"- {project['title']} ({project['years']}) – {project['grant_amount']} PLN\n"
        context += f"Źródło: {res['source']}\n\n"

    prompt = f"""
KONTEKST:
{context}

PYTANIE:
{query}

ODPOWIEDŹ:
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# === FUNKCJA: WEB ===
def answer_from_web(query):
    response = call_openai(query, use_web_search=True)
    if 'response' in response:
        return response['response']
    elif 'results' in response:
        return response['results'][0]
    else:
        return "Błąd lub brak odpowiedzi"

# === FUNKCJA: LLM-as-a-Judge ===
def judge_answers(question, rag_answer, web_answer):
    prompt = f"""
Masz pytanie i dwie odpowiedzi: jedna pochodzi z lokalnej bazy danych (RAG), druga z wyszukiwania w internecie (Web).

Oceń każdą odpowiedź w skali 0–2 w dwóch kategoriach:
- Factual Accuracy: Czy odpowiedź jest faktograficznie poprawna?
- Completeness: Czy odpowiedź jest kompletna i pokrywa całe pytanie?

Zwróć JSON w formacie:
{{
  "Factual Accuracy (RAG)": 0-2,
  "Completeness (RAG)": 0-2,
  "Factual Accuracy (Web)": 0-2,
  "Completeness (Web)": 0-2
}}

PYTANIE:
{question}

ODPOWIEDŹ RAG:
{rag_answer}

ODPOWIEDŹ WEB:
{web_answer}
""".strip()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    try:
        return json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        return None

# === ZAPIS DO EXCELA ===
def save_to_excel(question, rag_answer, web_answer, evaluation=None):
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    new_row = {
        "Pytanie": question,
        "Odpowiedź RAG": rag_answer,
        "Odpowiedź Web": web_answer,
        "Factual Accuracy (RAG)": "",
        "Completeness (RAG)": "",
        "Factual Accuracy (Web)": "",
        "Completeness (Web)": "",
        "Uwagi": ""
    }
    if evaluation:
        new_row.update(evaluation)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(EXCEL_PATH, index=False, engine="openpyxl")

# === GŁÓWNE URUCHOMIENIE ===
if __name__ == "__main__":
    if not os.path.exists(QUESTIONS_PATH):
        print(f"❌ Brak pliku z pytaniami: {QUESTIONS_PATH}")
        exit(1)

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    for idx, question in enumerate(questions):
        print(f"\n➡️ ({idx+1}/{len(questions)}) Pytanie: {question}")

        try:
            print("🔁 Generuję odpowiedź z RAG...")
            rag_answer = answer_from_rag(question)
            print("✅ RAG gotowe.")

            print("🌐 Generuję odpowiedź z Web Search...")
            web_answer = answer_from_web(question)
            print("✅ Web gotowe.")

            print("🧠 Oceniam jakość odpowiedzi...")
            evaluation = judge_answers(question, rag_answer, web_answer)
            print("📊 Ocena:", evaluation)

            save_to_excel(question, rag_answer, web_answer, evaluation)
            print("💾 Zapisano do Excela.")
        except Exception as e:
            print(f"❌ Błąd przy pytaniu: {question}\n{e}")
