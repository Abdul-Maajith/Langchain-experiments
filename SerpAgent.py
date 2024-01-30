from langchain.agents import Tool, create_react_agent, AgentExecutor
import os 
from langchain import hub
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

llm = ChatOpenAI(temperature=0)
search = DuckDuckGoSearchRun()
app = FastAPI()  

tools = [
    Tool(
        name="Google SERP Result",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

def get_answer(query: str):
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm, 
        tools,
        prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    response = agent_executor.invoke({"input": query})
    return response['output']

@app.get("/query", description="FactGPT")
async def qa(query: str):
    response = get_answer(query)
    return { "response": response }

@app.get("/", description="ping")
async def qa():
    return { "response": "Working" }