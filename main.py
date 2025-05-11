from fastapi import FastAPI
from fastapi import Query
from models import Scientist, ResearcherInfo, Affiliation, Keyword, Education, Publication
from models import RequestBody
from typing import List
from pathlib import Path
from fastapi import HTTPException
import json
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import openai


# Wczytaj zmienne środowiskowe
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Brakuje OPENAI_API_KEY w pliku .env")

openai.api_key = api_key

# Wczytaj dane o naukowcach
DATA_PATH = Path("data/scientist")
scientists: List[Scientist] = []

def read_all_scientists():
    if not DATA_PATH.exists():
        print(f"[ERROR] Folder '{DATA_PATH}' nie istnieje.")
        return

    files = list(DATA_PATH.glob("*.json"))
    if not files:
        print(f"[INFO] Brak plików JSON w folderze '{DATA_PATH}'")
        return

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                r = data.get("researcher", {})
                researcher = ResearcherInfo(
                    full_name=r.get("full_name"),
                    orcid_id=r.get("orcid_id"),
                    email=r.get("email"),
                    country=r.get("country"),
                    primary_affiliation=r.get("primary_affiliation"),
                )

                affiliations = [Affiliation(**a) for a in data.get("affiliations", [])]
                keywords = [Keyword(**k) for k in data.get("keywords", [])]
                education = [Education(**e) for e in data.get("education", [])]
                publications = [Publication(
                    title=p.get("title"),
                    journal=p.get("journal"),
                    doi=p.get("doi"),
                    year=p.get("year")
                ) for p in data.get("publications", [])]

                scientist = Scientist(
                    researcher=researcher,
                    affiliations=affiliations,
                    keywords=keywords,
                    education=education,
                    publications=publications
                )

                scientists.append(scientist)

        except Exception as e:
            print(f"[ERROR] Błąd w pliku {file.name}: {e}")

def get_earliest_year(affiliations: List[Affiliation]) -> Optional[int]:
    years = [
        int(a.start_date[:4])
        for a in affiliations
        if a.start_date and a.start_date[:4].isdigit()
    ]
    return min(years) if years else None


# FastAPI startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    read_all_scientists()
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Witaj w SARA API!"}

@app.get("/scientists", response_model=List[Scientist])
def search_scientists(
    affiliation: Optional[str] = Query(None, description="Nazwa instytucji"),
    keyword: Optional[str] = Query(None, description="Słowo kluczowe z profilu"),
    started_after: Optional[int] = Query(None, description="Kariera po roku"),
    orcid_id: Optional[str] = Query(None, description="ORCID ID"),
    full_name: Optional[str] = Query(None, description="Imię i nazwisko"),
    publication_keyword: Optional[str] = Query(None, description="Słowo z tytułu publikacji")
):
    results = scientists

    if orcid_id:
        results = [s for s in results if s.researcher.orcid_id == orcid_id]

    if full_name:
        results = [s for s in results if full_name.lower() in (s.researcher.full_name or "").lower()]

    if affiliation:
        results = [
            s for s in results if any(
                affiliation.lower() in (a.institution or "").lower()
                for a in s.affiliations
            )
        ]

    if keyword:
        results = [
            s for s in results if any(
                keyword.lower() in (k.keyword or "").lower()
                for k in s.keywords
            )
        ]

    if started_after is not None:
        results = [
            s for s in results
            if get_earliest_year(s.affiliations) and get_earliest_year(s.affiliations) > started_after
        ]

    if publication_keyword:
        results = [
            s for s in results if any(
                publication_keyword.lower() in (p.title or "").lower()
                for p in s.publications
            )
        ]

    return results


@app.get("/publications")
def get_publications(
    full_name: Optional[str] = Query(None),
    orcid_id: Optional[str] = Query(None)
):
    if not full_name and not orcid_id:
        raise HTTPException(status_code=400, detail="Podaj 'full_name' lub 'orcid_id'.")

    for scientist in scientists:
        if orcid_id and scientist.researcher.orcid_id == orcid_id:
            return [p.dict() for p in scientist.publications]

        if full_name and full_name.lower() in (scientist.researcher.full_name or "").lower():
            return [p.dict() for p in scientist.publications]

    raise HTTPException(status_code=404, detail="Nie znaleziono naukowca.")


@app.post("/request")
def handle_request(payload: RequestBody):
    if not payload.request:
        raise HTTPException(status_code=400, detail="Pole 'request' jest wymagane.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": payload.request}]
        )
        reply = response.choices[0].message["content"]
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/sara_request")
def handle_sara_request(payload: RequestBody):
    if not payload.request:
        raise HTTPException(status_code=400, detail="Pole 'request' jest wymagane.")

    try:
        return {"response": payload.request}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
