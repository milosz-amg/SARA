import os
import openai
import pandas as pd
from dotenv import load_dotenv

# === Ładujemy klucz API z .env ===
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Pobierz imię i nazwisko ===
name_input = input("Podaj imię i nazwisko naukowca (np. Patryk Żywica): ").strip()

def get_inflected_cases(full_name):
    prompt = f"""
Odmień imię i nazwisko "{full_name}" przez następujące przypadki liczby pojedynczej:
- Mianownik
- Dopełniacz
- Biernik
- Narzędnik

Zwróć wynik jako JSON w formacie:
{{
  "mianownik": "...",
  "dopełniacz": "...",
  "biernik": "...",
  "narzędnik": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    content = response.choices[0].message.content.strip()

    # ✅ Usuń znaczniki Markdown, jeśli są obecne
    if content.startswith("```"):
        content = content.strip("`").split("\n", 1)[1]  # usuwa pierwszą linię (```json)
        content = content.rsplit("```", 1)[0].strip()   # usuwa końcową ```

    try:
        cases = eval(content)
        if not isinstance(cases, dict):
            raise ValueError("Niepoprawny format JSON")
        return cases
    except Exception as e:
        print(f"❌ Błąd przy parsowaniu odpowiedzi GPT: {e}")
        print("Odpowiedź GPT (raw):", response.choices[0].message.content)
        exit(1)


# === Pobieramy odmiany ===
cases = get_inflected_cases(name_input)

# === Generujemy pytania ===
templates = [
    f"Jakie są główne obszary badawcze {cases['dopełniacz']}?",
    f"Jakie projekty badawcze realizował(a) {cases['mianownik']} w latach 2020–2025?",
    f"Na jakie granty aplikował(a) {cases['mianownik']} w latach 2019–2024?",
    f"Jakie są najczęściej cytowane publikacje {cases['dopełniacz']}?",
    f"Z kim najczęściej współpracuje {cases['mianownik']}?",
    f"Które publikacje {cases['dopełniacz']} były współautorstwem z naukowcami spoza UAM?",
    f"Jakie były najnowsze projekty realizowane przez {cases['biernik']} w 2023 roku?",
    f"Czy {cases['mianownik']} zajmuje się sztuczną inteligencją lub jej zastosowaniami?",
    f"Jakie funkcje pełnił(a) {cases['mianownik']} na UAM?",
    f"Kto współtworzył publikacje z {cases['narzędnik']} w latach 2020–2023?"
]

templates += [
    "Którzy naukowcy z UAM prowadzą badania nad sztuczną inteligencją?",
    "Którzy naukowcy z WMiI UAM rozpoczęli karierę po 2017 roku?",
    "Jakie projekty badawcze związane z AI były realizowane na UAM w 2023 roku?",
    "Pokaż projekty związane z logiką rozmytą o budżecie powyżej 50 000 PLN.",
    "Którzy naukowcy złożyli wnioski o grant NCN OPUS 24 w 2023 roku?",
    "Kto jest najczęściej cytowanym naukowcem z WMiI UAM?",
    "Jakie projekty AI w medycynie były realizowane w Polsce w 2024 roku?",
    "Jakie granty AI oferują dofinansowanie na poziomie 95% w modelu ryczałtowym?",
    "Które granty AI przewidują uczestnictwo państw EOG (Norwegia, Islandia, Liechtenstein)?"
]

# === Zapisz do pliku ===
parts = name_input.lower().split()
imie = parts[0]
nazwisko = "_".join(parts[1:]) if len(parts) > 1 else "nazwisko"
# === Przygotuj folder "ankiety" i ścieżkę ===
os.makedirs("ankiety", exist_ok=True)
filename = f"{imie}_{nazwisko}_sara.xlsx"
filepath = os.path.join("ankiety", filename)

# === Zapisz plik ===
df = pd.DataFrame({"Pytanie": templates})
df.to_excel(filepath, index=False, engine="openpyxl")

print(f"\n✅ Zapisano plik z pytaniami: {filepath}")

print(f"\n✅ Zapisano plik z pytaniami: {filename}")
