import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool


load_dotenv()

search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

tools = [
    Tool(
        name="TavilySearch",
        func=search.run,
        # description="Use this tool to search for recent or factual information from the web" # for responses in english
        description="Użyj tego narzędzia, aby wyszukać najnowsze lub faktograficzne informacje z internetu. Odpowiadaj po polsku."
    )
]

def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-large")
