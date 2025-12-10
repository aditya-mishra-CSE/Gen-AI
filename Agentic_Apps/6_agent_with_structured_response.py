"""
Problem Statement:
    - LLM responses are uncertain.

    - We need a specific format to deliver the best user experience.

Solution:
    - we can use Pydantic modeling for structured responses in NLP.

"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

load_dotenv()

class MailResponseModel(BaseModel):
    subject: str
    body: str

class ItinenaryResponseModel(BaseModel):
    day1: str
    day2: str
    day3: str

llm_model = ChatGroq(
    model='llama-3.3-70b-versatile',
    max_tokens=200
)

agent = create_react_agent(
    model=llm_model,
    tools=[],
    prompt="You're a helpful assistant",
    # response_format= ItinenaryResponseModel
    response_format= MailResponseModel
)

def extractResponse(response):
    print(response["messages"][-1].content) 

def getResponseFromAgent():
    query = HumanMessage(input("Enter a query:"))

    response = agent.invoke({
        "messages": [query]
    })

    extractResponse(response)

getResponseFromAgent()