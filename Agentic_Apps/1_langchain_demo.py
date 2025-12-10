from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI

# from langchain_groq import ChatGroq

from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq")

# model = ChatOpenAI(model="gpt-4o-mini")

# model = ChatGroq(model="llama-3.3-70b-versatile")

response = model.invoke("Who is dhoni?")

print(response.content)