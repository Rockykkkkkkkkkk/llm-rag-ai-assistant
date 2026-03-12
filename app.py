from fastapi import FastAPI
from src.rag_pipeline import build_rag_pipeline

app = FastAPI()

rag = None

@app.get("/")
def home():
    return {"message": "LLM RAG AI Assistant"}

@app.post("/ask")
def ask(question: str):
    response = rag.run(question)
    return {"answer": response}
