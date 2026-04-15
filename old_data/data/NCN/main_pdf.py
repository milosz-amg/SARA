import pdfplumber
import os
import re
import json

pdf_folder = "./pdf_doc"  # folder z PDF
output_json = "./pdf_doc_rag.json"

documents = []

# Regexy
project_start_regex = re.compile(r'^\s*(\d+)\.\s*(HS\d+|NZ\d+)?\s*(.*)$')  # LR, panel opcjonalny, reszta
kwota_regex = re.compile(r'(\d{3,}(?:\s\d{3})*)')  # liczba jako kwota

for pdf_file in os.listdir(pdf_folder):
    if not pdf_file.endswith(".pdf"):
        continue
    pdf_path = os.path.join(pdf_folder, pdf_file)

    # Wszystkie strony w jeden strumień linii
    all_lines = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                page_lines = [line.strip() for line in text.split("\n") if line.strip()]
                all_lines.extend(page_lines)

    buffer = ""
    for line in all_lines:
        match_start = project_start_regex.match(line)
        if match_start:
            # Nowy projekt → przetwarzamy poprzedni bufor
            if buffer:
                # Wyodrębnianie pól
                match_buf = project_start_regex.match(buffer)
                if match_buf:
                    LR, panel, rest = match_buf.groups()
                    kwota_match = kwota_regex.search(rest)
                    kwota = kwota_match.group(0).replace(" ", "") if kwota_match else ""
                    if kwota_match:
                        idx = kwota_match.start()
                        before_kwota = rest[:idx].strip()
                        after_kwota = rest[kwota_match.end():].strip()
                    else:
                        before_kwota = rest
                        after_kwota = ""

                    opis_pl = before_kwota if before_kwota else "-"
                    tytul_en = after_kwota if after_kwota else "-"

                    text_field = (
                        f"Opis PL: {opis_pl}\n"
                        f"Tytuł EN: {tytul_en}\n"
                        f"Przyznane finansowanie: {kwota}\n"
                        f"Panel: {panel if panel else '-'}\n"
                        f"LR: {LR}"
                    )

                    doc = {
                        "text": text_field,
                        "metadata": {
                            "LR": LR,
                            "panel": panel if panel else "-",
                            "opis_pl": opis_pl,
                            "kwota": kwota,
                            "tytul_en": tytul_en,
                            "pdf_file": pdf_file
                        }
                    }
                    documents.append(doc)
            buffer = line  # zaczynamy nowy projekt
        else:
            # Linia kontynuacji → doklejamy do bufora
            buffer += " " + line

    # Przetwarzamy ostatni projekt w buforze
    if buffer:
        match_buf = project_start_regex.match(buffer)
        if match_buf:
            LR, panel, rest = match_buf.groups()
            kwota_match = kwota_regex.search(rest)
            kwota = kwota_match.group(0).replace(" ", "") if kwota_match else ""
            if kwota_match:
                idx = kwota_match.start()
                before_kwota = rest[:idx].strip()
                after_kwota = rest[kwota_match.end():].strip()
            else:
                before_kwota = rest
                after_kwota = ""

            opis_pl = before_kwota if before_kwota else "-"
            tytul_en = after_kwota if after_kwota else "-"

            text_field = (
                f"Opis PL: {opis_pl}\n"
                f"Tytuł EN: {tytul_en}\n"
                f"Przyznane finansowanie: {kwota}\n"
                f"Panel: {panel if panel else '-'}\n"
                f"LR: {LR}"
            )

            doc = {
                "text": text_field,
                "metadata": {
                    "LR": LR,
                    "panel": panel if panel else "-",
                    "opis_pl": opis_pl,
                    "kwota": kwota,
                    "tytul_en": tytul_en,
                    "pdf_file": pdf_file
                }
            }
            documents.append(doc)

# Zapis JSON
os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=2)

print(f"Zapisano {len(documents)} projektów do: {output_json}")
