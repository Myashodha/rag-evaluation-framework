from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rag import RAGPipeline


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str]


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rag = pipeline or RAGPipeline()
        app.state.rag.build_index()          # indexing happens once, at startup
        yield

    app = FastAPI(title="GubbisCart RAG API", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": app.state.rag.collection.count()}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        try:
            out = app.state.rag.ask(req.question)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {str(e)[:200]}")
        return AskResponse(answer=out["answer"], sources=out["sources"], contexts=out["contexts"])

    return app


app = create_app()