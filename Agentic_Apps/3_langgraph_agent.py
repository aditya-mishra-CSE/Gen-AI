from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

def displayConversation(response):
    for message in response["messages"]:
        if isinstance(message, AIMessage):
            print("\n\n\nAssistant:")
        else:
            print("\n\n\nUser:")
        
        print(message.content)

model = ChatGroq(model='llama-3.3-70b-versatile', max_tokens=100)

agent = create_react_agent(
        model,
        tools=[], 
        prompt="You are a helpful assistant")

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]}
)

displayConversation(response)

