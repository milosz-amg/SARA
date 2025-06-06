from utils.azure_openai import get_llm

llm = get_llm()

response = llm.invoke("Kim jest Patryk Żywica? Pracownik UAM")

print(response)
