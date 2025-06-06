from dotenv import load_dotenv
from utils.azure_openai import get_llm, tools
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType

load_dotenv()


llm = get_llm()

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

query = "Kim jest Patryk Żywica pracownik UAM? Z jakimi naukowcami mógłby współpracować, podaj przykładowych naukowców?"

# response = llm.invoke("Kim jest Patryk Żywica? Pracownik UAM") # pure llm without any tools for web searching
response = agent.invoke(query)

print(response)
