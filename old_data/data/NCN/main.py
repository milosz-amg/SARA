import json
import os

# Ścieżka do pliku JSON
input_path = "./listy_rankingowe.json"
output_path = "./projekty_rag.json"

# Wczytanie danych
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for projekt in data["listy_rankingowe"]:
    # Budowanie tekstu dokumentu
    doc = ""
    doc += f"Tytuł PL: {projekt.get('tytul.pl', '')}\n"
    doc += f"Tytuł EN: {projekt.get('tytul.en', '')}\n"
    doc += f"Uczelnia: {projekt.get('jednostka.poziom1', '')}, {projekt.get('jednostka.poziom2', '')}\n"
    kierownik = projekt.get('kierownik', {})
    doc += f"Kierownik: {kierownik.get('tytul','')} {kierownik.get('imie','')} {kierownik.get('imie2','')} {kierownik.get('nazwisko','')}\n"
    doc += f"Koszt: {projekt.get('koszt','')}\n"
    doc += f"Data aktualizacji: {projekt.get('dataaktualizacji','')}\n"
    
    # Opcjonalnie: podział na fragmenty jeśli dokument jest długi
    # Tutaj przyjmujemy, że projekt mieści się w jednym fragmencie
    
    documents.append({
        "text": doc.strip(),
        "metadata": {
            "id": projekt.get("id"),
            "edycja": projekt.get("edycja"),
            "typ_konkursu": projekt.get("typkonkursu"),
            "panel": projekt.get("panel")
        }
    })

# Zapis gotowego JSON do użycia w RAG
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"Gotowe dokumenty do RAG zapisane w: {output_path}")
