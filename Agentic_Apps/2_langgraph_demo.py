from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langgraph.graph import START, END, StateGraph

from langgraph.graph.message import add_messages, Annotated

from typing_extensions import TypedDict

load_dotenv()

# Step-1: Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Step-2: Create the StateGraph instance and pass the state
graph_builder = StateGraph(State)


# Step-3: Define the Nodes
llm_model = ChatGroq(model='llama-3.3-70b-versatile')

def chatbotNode(state:State):
    response = llm_model.invoke(state["messages"])

    return {"messages": response}

graph_builder.add_node(chatbotNode)


# Step-4: Define the Edges
graph_builder.add_edge(START, "chatbotNode")
graph_builder.add_edge("chatbotNode", END)


# Step-5: compile the graph
graph = graph_builder.compile()


# graph visualization
def visualize_graph():
    try:
        img = graph.get_graph().draw_mermaid_png()

        with open("langgraph_demo.png", "wb") as fp:
            fp.write(img)

    except Exception:
        pass

# streaming
def stream_graph_updates(user_input:str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"].content)

# visualize_graph()

while True:
    try:
        user_input = input("Enter the query:")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Good bye !")
            break
    
        stream_graph_updates(user_input=user_input)
    except Exception as ex:
        print("Invalid input" + ex.message)