from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# App init
app = FastAPI(title="SARA")

# Load scientists
def load_scientists(filename="scientists.json"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading scientists: {e}")
        return []

scientists = load_scientists()

# ------------------------------
#          MODELE
# ------------------------------

class Grant(BaseModel):
    name: str
    year: int

class Award(BaseModel):
    name: str
    year: int

class Publication(BaseModel):
    title: str
    year: int

class Scientist(BaseModel):
    id: int
    name: str
    field: str
    start_year: int
    description: str
    current_institution: str
    institutions: List[str]
    publications: List[Publication]
    awards: List[Award]
    grants: List[Grant]

class RequestBody(BaseModel):
    request: str

# ------------------------------
#         ENDPOINTS
# ------------------------------

@app.get("/")
def home():
    return {"message": "Welcome SARA!"}

@app.get("/scientists", response_model=List[Scientist])
def get_all_scientists():
    return scientists

@app.get("/scientists/{scientist_id}", response_model=Scientist)
def get_scientist(scientist_id: int):
    for s in scientists:
        if s["id"] == scientist_id:
            return s
    raise HTTPException(status_code=404, detail="Scientist not found")

@app.get("/scientists/name/{name}", response_model=Scientist)
def get_scientist_by_name(name: str):
    for s in scientists:
        if s["name"].lower() == name.lower():
            return s
    raise HTTPException(status_code=404, detail="Scientist not found")

@app.get("/scientists/{scientist_id}/grants", response_model=List[Grant])
def get_scientist_grants(scientist_id: int):
    for s in scientists:
        if s["id"] == scientist_id:
            return s.get("grants", [])
    raise HTTPException(status_code=404, detail="Scientist not found")

@app.get("/scientists/{scientist_id}/description")
def get_description(scientist_id: int):
    for s in scientists:
        if s["id"] == scientist_id:
            return {"description": s.get("description", "")}
    raise HTTPException(status_code=404, detail="Scientist not found")

@app.get("/scientists/{scientist_id}/current_institution")
def get_current_institution(scientist_id: int):
    for s in scientists:
        if s["id"] == scientist_id:
            return {"current_institution": s.get("current_institution", "")}
    raise HTTPException(status_code=404, detail="Scientist not found")


# This endpoint will be usefull when we will build AI agent :)
@app.post("/request")
def handle_request(payload: RequestBody):
    if not payload.request:
        raise HTTPException(status_code=400, detail="Missing 'request' field")
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": payload.request}]
        )
        reply = response.choices[0].message.content
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
