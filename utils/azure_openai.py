import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings

load_dotenv()

def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")
