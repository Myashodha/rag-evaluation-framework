from fastapi.testclient import TestClient

from src.api import create_app
from src.rag import RAGPipeline


class StubLLM:
    def invoke(self, prompt):
        class R:
            content = "Customers can request a refund within 30 days of delivery."
        return R()


def make_client(embedder=None):
    rag = RAGPipeline(llm=StubLLM(), collection_name="api_test", embedder=embedder)
    return TestClient(create_app(rag))


def test_health_and_ask():
    with make_client() as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/ask", json={"question": "What is the refund period?"})
        body = r.json()
        assert r.status_code == 200
        assert body["answer"].strip()
        assert len(body["contexts"]) > 0
        assert body["sources"]


def test_rejects_empty_question():
    with make_client() as client:
        assert client.post("/ask", json={"question": ""}).status_code == 422