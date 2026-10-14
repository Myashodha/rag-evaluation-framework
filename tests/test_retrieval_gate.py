import pytest

from src.dataset import load_golden
from src.rag import RAGPipeline

ROWS = [r for r in load_golden() if r.answerable]


@pytest.fixture(scope="module")
def retriever():
    p = RAGPipeline(llm=object(), collection_name="gate")
    p.build_index()
    return p


def test_hit_rate_at_3(retriever):
    hits = sum(
        any(x["source"] in r.expected_sources for x in retriever.retrieve(r.question))
        for r in ROWS
    )
    rate = hits / len(ROWS)
    assert rate >= 0.90, f"hit rate fell to {rate:.2f}"