# from fastapi import FastAPI, HTTPException, Query
# from pydantic import BaseModel
# from sara_assistant import call_openai_json
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # (opcjonalnie) CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     prompt: str
#     model: str = "gpt-4o"
#     temperature: float = 1.0
#     use_web_search: bool = False

# @app.post("/query")
# async def query_openai(request: QueryRequest, use_web_search: bool = Query(False)):
#     try:
#         result = call_openai_json(
#             prompt=request.prompt,
#             model=request.model,
#             temperature=request.temperature,
#             use_web_search=use_web_search
#         )
#         return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# main.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from azure_client import ask_azure_openai  # ← import funkcji

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"
    temperature: float = 1.0
    use_web_search: bool = False  # możesz rozbudować to później

@app.post("/query")
async def query_openai(request: QueryRequest):
    try:
        result = ask_azure_openai(
            prompt=request.prompt,
            model=request.model,
            temperature=request.temperature
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
