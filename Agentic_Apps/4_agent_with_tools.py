from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessageChunk
from langchain.tools import tool
import os

load_dotenv()

# ================== tools ==================
@tool
def addFile(path:str, filename:str) -> str:
    """Create a new file in the given path"""

    filepath = os.path.join(path, filename)

    if not os.path.exists(filepath):
        with open(filepath, 'w') as fp:
            fp.write("New file")
        
        print(f"File: {filename} is created")
 
    else:
        print(f"File already exists")

@tool
def addFolder(path:str, dirname:str) -> str:
    """Create a new directory in the given path"""

    dirpath = os.path.join(path, dirname)
    
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        
        print(f"Directory with {dirname} is created")
    
    else:
        print(f"Directory already exists")

@tool
def removeFile(path:str, filename:str) -> str:
    """Remove a new file in the given path"""

    filepath = os.path.join(path, filename)

    if os.path.exists(filepath):
        os.remove(filepath)

        print(f"File: {filename} is deleted")
 
    else:
        print(f"File doesn't exists")

@tool
def removeFolder(path:str, dirname:str) -> str:
    """Remove a directory in the given path"""

    dirpath = os.path.join(path, dirname)
    
    if os.path.exists(dirpath):
        os.rmdir(dirpath)
        
        print(f"Directory with {dirname} is deleted")
    
    else:
        print(f"Directory doesn't exists")

@tool
def renameItem(path:str, oldname:str, newname:str) -> str:
    """Rename a directory in the given path"""

    oldpath = os.path.join(path, oldname)

    newpath = os.path.join(path, newname)

    print("old path:" + oldpath)

    print("new path:" + newpath)
    
    if os.path.exists(oldpath):
        os.rename(oldpath, newpath)
        
        print(f"Directory with {oldname} is successfully renamed")
    
    else:
        print(f"Directory doesn't exists")

# ================== Message History ==================

user_conversation_history = [
    SystemMessageChunk("You are a helpful assistant")
]

# ================== Agent: LLM + Tools ==================

model = ChatGroq(model='llama-3.3-70b-versatile', max_tokens=200, max_retries=2)

agent = create_react_agent(
        model,
        tools=[
                addFile, addFolder, removeFile, 
                removeFolder, renameItem
            ], 
        prompt=None)

# ================== Helper Functions ==================

def displayConversation(conversation):
    for message in conversation:
        if isinstance(message, AIMessage):
            print("\n\nAssistant:")
        elif isinstance(message, SystemMessageChunk):
            print("\n\nSystem:")
        else:
            print("\n\nUser:")
        
        print(message.content)

# graph visualization
def visualize_graph():
    try:
        img = agent.get_graph().draw_mermaid_png()

        with open("langgraph_with_tools_demo.png", "wb") as fp:
            fp.write(img)

    except Exception:
        pass

# ================== User Interface ==================

while True:
    user_input = input("Enter the query:").lower()

    if user_input in ["quit", "q", "exit"]:
        print("Good bye")

        break

    user_conversation_history.append(HumanMessage(user_input))

    response = agent.invoke(
        {"messages": user_conversation_history}
    )

    user_conversation_history.append(response["messages"][-1])

    displayConversation(user_conversation_history)

visualize_graph()