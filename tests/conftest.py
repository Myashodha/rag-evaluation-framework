from _pytest.legacypath import FixtureRequest_fspath
import pytest
from src.rag import RAGPipeline

@pytest.fixture(scope = "session")
def rag():
    pipeline=RAGPipeline()
    pipeline.build_index()
    return pipeline

