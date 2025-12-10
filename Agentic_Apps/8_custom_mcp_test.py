import asyncio
import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

# class TokenAudit(BaseCallbackHandler):
#     def on_llm_end(self, response, *, run_id=None, parent_run_id=None, tags=None, **kwargs):
#         usage = {}
#         # LC <=0.1 / <=0.2 variants
#         # Try llm_output
#         if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
#             usage = response.llm_output.get("token_usage") or response.llm_output.get("usage") or {}
#         # Try additional_kwargs on messages (provider-specific)
#         if not usage and getattr(response, "generations", None):
#             try:
#                 meta = response.generations[0][0].message.response_metadata
#                 usage = meta.get("token_usage") or meta.get("usage") or meta
#             except Exception:
#                 pass
#         print(f"[USAGE] {usage}") 

async def run_agent():
    llm_model = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=50, 
        # callbacks=[TokenAudit()]
    )

    # accessing my custom mcp server locally
    client = MultiServerMCPClient(
        {
            "MyFileSystem": {
                "command": "python", 
                "args":[
                    "./8_filesystem_mcp.py"
                ], 
                "transport":"stdio"
            }
        }
    )

    tools = await client.get_tools()

    # print(tools)

    agent = create_react_agent(
        model=llm_model, 
        tools=tools
    )

    try:
        query = HumanMessage(input("You:"))

        state = await agent.ainvoke({
            "messages": [query]
        })

        print(state["messages"][-1].content)

    except Exception as ex:
        print(f"Exception:{ex}")


if __name__ == "__main__":
    asyncio.run(run_agent())