from langchain_core.vectorstores import InMemoryVectorStore
from utils.azure_openai import get_embeddings

def get_vector_store():
    embeddings = get_embeddings()
    return InMemoryVectorStore(embeddings)
