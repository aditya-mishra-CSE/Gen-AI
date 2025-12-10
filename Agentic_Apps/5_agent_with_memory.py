"""
Problem 1: Context Forgetting Problem
- If we dont pass the context, LLM cannot answer the followups.


Problem 2: Drawback of storing user conversation history(Token Limit Reached)

- if entire conversation history is stored as list and passed to the LLM for context,
we are consuming more tokens ---> This is not cost effective.

- we can do sliding window, but still problem exists.

Compromising on Cost for the Context.

----------------

Solution(Not Ideal but better):
-> short term memory for the agent
    - remember previous interactions as summary(or selected history or last n steps),
            within a single thread or conversation.

    - scope: Thread-scoped

    examples:
    - using RAM(inmemorysaver)
    - using persistant memory(postgressaver, mongodbsaver,...)

-> long term memory for the agent
    - retain the information across different conversations or sessions.

    - scope: Custom namespace scope

    examples:
    - InMemoryStore(RAM)
    - DBStore(persistance)

"""


from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

checkpoint = InMemorySaver()

llm_model = ChatGroq(
    model = "llama-3.3-70b-versatile", 
    max_tokens = 100
)

agent = create_react_agent(
 model=llm_model,
 tools=[],
 checkpointer=checkpoint,
 prompt="You are a helpful assistant"
)

checkpoint_config1 = {
    "configurable": {
        "thread_id": 1
    }
}

# same thread_id refers to the same conversation reference
# different thread_id refers to the different conversations.
checkpoint_config2 = {
    "configurable": {
        "thread_id": 1
    }
}

def extractResponse(response):
    print(response["messages"][-1].content)

query1 = HumanMessage(input("Enter the query:"))

# ainvoke : asynchronous invoke
response1 = agent.invoke(
    {"messages": [query1]}, 
    config=checkpoint_config1
)

extractResponse(response1)

query2 = HumanMessage(input("Enter the follow-up:"))


response2 = agent.invoke(
    {
        "messages": [query2]
    },
    config=checkpoint_config2
)

extractResponse(response2)