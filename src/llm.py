import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(provider : str | None =None):
    provider =provider or os.getenv("LLM_PROVIDER","ollama")
    if provider == "groq":
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
        )
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=0,
        num_ctx=4096,
    )


