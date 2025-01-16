from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd


df = pd.read_csv("../data/df_rent.csv")

agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model="gpt-4o-mini"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# agent.invoke("Quantas linhas existem no conjunto de dados?")
response = agent.invoke("Qual o maior valor de preço no distrito Belém/São Paulo?")
print(response["output"])
