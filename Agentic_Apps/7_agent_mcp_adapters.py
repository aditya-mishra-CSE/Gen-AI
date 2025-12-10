import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

"""
MCP_Adapaters in Langchain:
1. Host : application
2. MCP Client : instance/process that communicates with the MCP server
3. MCP Server : remote/local server that serves the MCP client requests.

"""

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

llm_model1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=100,
    max_retries=2
)


llm_model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    # max_tokens=100
)

def extractResponse(response):
    print(response["messages"][-1].content)


async def run_agent():
    # defining the clients
    client = MultiServerMCPClient(
        {
            # local MCP server that communciates with remote
            "github": {
               "command": "npx",
               "args": [
                   "-y",
                   "@modelcontextprotocol/server-github"
               ],
               "env": {
                   "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN
               },
               "transport": "stdio"
           },
            "filesystem": {
               "command": "npx",
               "args": [
                   "-y",
                   "@modelcontextprotocol/server-filesystem",
                   "/Users/vg588/Desktop/Gen-AI/lang-chain/simple-chatbot"
               ],
               "transport":"stdio"
           }
            # "github": {
            #         "command": "docker",
            #         "args": [
            #         "run",
            #         "-i",
            #         "--rm",
            #         "-e",
            #         "GITHUB_PERSONAL_ACCESS_TOKEN",
            #         "ghcr.io/github/github-mcp-server"
            #     ],
            #     "env": {
            #         "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN
            #     }, 
            #     "transport": 'stdio'
            # }
        }
    )

    tools = await client.get_tools()

    agent = create_react_agent(
        model=llm_model1,
        tools=tools
    )

    while True:
        user_input = input("Enter a query:")

        if user_input in ["quit", "exit", "q"]:
            break
        
        query = HumanMessage(user_input)

        try:
            # ainvoke: asynchronous invoke
            response = await agent.ainvoke({
                "messages": [query]
            })
            
        except Exception as ex:
            print(f"Exception: {ex}")

        print(extractResponse(response))

if __name__=="__main__":
    asyncio.run(run_agent())