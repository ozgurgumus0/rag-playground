"""Smoke test: checks that your keys work and LangChain can talk to Gemini."""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Read the keys from .env into environment variables.
# LangChain picks up GOOGLE_API_KEY (and the LANGSMITH_* settings) automatically.
load_dotenv()

# 1. Chat model: ask a question and print the answer.
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")
reply = llm.invoke("In one sentence, what is RAG?")
print("Chat reply:", reply.text)

# 2. Embedding model: turn text into a list of numbers (a vector).
# RAG uses these vectors to find text that is similar in meaning.
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vector = embeddings.embed_query("hello world")
print("Embedding length:", len(vector))
