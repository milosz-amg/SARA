import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEMANTIC_SCHOLAR_API_URL = os.getenv("SEMANTIC_SCHOLAR_API_URL", "https://api.semanticscholar.org/graph/v1")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

DEFAULT_TIMEOUT = float(os.getenv("SS_TIMEOUT", "20"))
DEFAULT_LIMIT = int(os.getenv("SS_LIMIT", "25"))
DEFAULT_FIELDS_AUTHOR = os.getenv(
    "SS_AUTHOR_FIELDS",
    "authorId,name,url,affiliations,homepage,paperCount,citationCount,hIndex"
).split(",")