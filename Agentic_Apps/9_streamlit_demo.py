import asyncio
import os
import uuid
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from google import genai

# gemini_client = genai.client()

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

os.environ["GRPC_VERBOSITY"] = "ERROR"

os.environ["GLOG_minloglevel"] = "2"

st.title("ChatGPT clone")

model_choices = [
    {"key": "Groq", "label": "Llama 3.3"},
    {"key": "Gemini-Chat", "label": "Gemini 2.0"},
    # {"key": "Gemini-Image", "label": "Gemini 2.0 Image Generation"}
]

llm_models = {
    "Groq": ChatGroq(model="llama-3.3-70b-versatile", temperature=0),
    "Gemini-Chat": ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
    # "Gemini-Image": ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp-image-generation", temperature=0)
    # "Gemini": gemini_client
}

selected = st.selectbox(
    "Select a model:",
    options=model_choices,
    format_func=lambda model: model["key"],
    width=200
)

st.session_state.selected_model = llm_models[selected["key"]]

def get_default_chat_template()->str:
    return {
            "title": None,
            "conversation_id": str(uuid.uuid4()),
            "checkpointer_instance": MemorySaver(), 
            "messages": []
        }

if ("chats" not in st.session_state):
    st.session_state.chats = [get_default_chat_template()]

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = st.session_state.chats[0]["conversation_id"]

def display_chat_history(chat):
    for role, message in chat["messages"]:
        with st.chat_message(role):
            st.markdown(message)

async def stream_graph_updates(user_input: str, chat, image_file=None) -> str:
    """ Updated to optionally receive image_file for Gemini-Image """
    assistant_response = ""

    try:
        # message_placeholder = st.empty()

        # Choose model:
        if image_file:
            # Use Gemini-Image
            model = st.session_state.selected_model

            image_bytes = image_file.read()

            image_dict = {
                "type": "image",
                "data": image_bytes,
                "mime_type": image_file.type
            }

            # Send image bytes to Gemini-Image model
            result = model.invoke([image_dict])  # adapt this to actual API (.run/.invoke/etc.)
            
            assistant_response = str(result)
            
            # message_placeholder.markdown(assistant_response)
        else:
            # Use text model (Groq or Gemini-Chat)
            chat_config = {
                "configurable": {"thread_id": chat["conversation_id"]}
            }

            events = st.session_state.agent.stream(
                {"messages": [HumanMessage(user_input)]},
                config=chat_config
            )
            
            for event in events:
                for value in event.values():
                    content = value["messages"][-1].content
                    
                    assistant_response += content
                    
                    # message_placeholder.markdown(assistant_response)
        
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        chat["messages"].append(("assistant", assistant_response or "_(no output)_"))
    
    except Exception as ex:
        st.error(f"Exception: {ex}")
    finally:
        return "done"

def create_new_chat():
    st.session_state.chats.insert(0, get_default_chat_template())

    st.session_state.active_chat_id = st.session_state.chats[0]["conversation_id"]

async def update_active_chat(chat):
    st.session_state.active_chat_id = chat["conversation_id"]

    await load_agent(get_chat(st.session_state.active_chat_id))

def get_chat(chat_id):
    filtered = [chat for chat in st.session_state.chats 
                    if chat["conversation_id"] == chat_id]

    if filtered:
        return filtered[0]

    if st.session_state.chats:
        return st.session_state.chats[0]
    
    else:
        create_new_chat()

        return st.session_state.chats[0]


@st.dialog("Delete Confirmation")
def delete_chat(chat_id):
    st.write("Do you want to delete the chat?")

    if st.button("confirm"):
        st.session_state.chats = [chat for chat in st.session_state.chats 
                                    if chat["conversation_id"] != chat_id]
        
        st.rerun()

async def load_tools():
    client = MultiServerMCPClient({
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
    })

    tools = await client.get_tools()

    return tools

async def load_agent(chat):
    llm_model = st.session_state.selected_model

    # agent_tools = await load_tools()
    agent_tools = []

    # print("agent_tools:", agent_tools)

    st.session_state.agent = create_react_agent(
        model=llm_model,
        tools=agent_tools,
        checkpointer= chat["checkpointer_instance"],
        prompt="You are a helpful assistant."
    )

async def run_agent():
    print(st.session_state.selected_model)

    with st.sidebar:
        st.button("new chat", on_click=create_new_chat, width="stretch")

        for chat in st.session_state.chats:
            with st.container(horizontal=True, vertical_alignment="center"):
                [col1, col2] = st.columns([4, 1])

                chat_title = (chat["title"] if chat["title"] else chat["conversation_id"])

                with col1:
                    st.button(chat_title, 
                                    on_click=update_active_chat, 
                                    args=[chat], 
                                    width="stretch")

                with col2:
                    st.button('', key=chat["conversation_id"],
                        icon=":material/delete:", 
                        type="primary",
                        on_click=delete_chat, args=[chat["conversation_id"]])

    active_chat = get_chat(st.session_state.active_chat_id)

    display_chat_history(active_chat)

    await load_agent(active_chat)

    prompt = st.chat_input("How can I help you today?", accept_file=True)

    # TEXT chat
    if prompt and prompt.text:
        with st.chat_message("user"):
            st.markdown(prompt.text)

        if active_chat["title"] == None:
            active_chat["title"] = prompt.text[:30]

        active_chat["messages"].append(("user", prompt.text))

        with st.spinner():
            await stream_graph_updates(prompt.text, active_chat)

    # IMAGE chat
    if prompt and prompt["files"]:
        image_file = prompt["files"][0]

        st.image(image_file)

        if active_chat["title"] == None:
            active_chat["title"] = "Image upload"

        active_chat["messages"].append(("user", f"Sent image: {image_file.name}"))

        with st.spinner():
            await stream_graph_updates("", active_chat, image_file=image_file)

asyncio.run(run_agent())
