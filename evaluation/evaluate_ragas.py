import sys, json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.dataset import load_golden
from src.llm import get_llm

try:
    from langchain_community.chat_models import vertexai as _lc_vertexai  # noqa
except ImportError:
    from langchain_google_vertexai import ChatVertexAI as _ChatVertexAI
    _shim = types.ModuleType("langchain_community.chat_models.vertexai")
    _shim.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _shim


import pandas as pd

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,           # fixed: was ResponseRelevency (typo)
)
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings



LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None


def main():
    df = pd.read_csv(ROOT / "reports" / "rag_outputs.csv")   # fixed: missing .csv
    truth = {r.id: r.ground_truth for r in load_golden()}
    df["ground_truth"] = df["id"].map(truth)
    if LIMIT:
        df = df.head(LIMIT)

    # contexts column is stored as a JSON string — parse it back to a list
    samples = [
        SingleTurnSample(
            user_input=row.question,
            response=row.answer,                              # fixed: was 'answer'
            retrieved_contexts=json.loads(row.contexts),     # fixed: was raw string
            reference=row.ground_truth,                      # fixed: was 'ground_truth'
        )
        for row in df.itertuples()
    ]

    dataset = EvaluationDataset(samples=samples)

    judge = LangchainLLMWrapper(get_llm("groq"))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=2, timeout=100),
    )

    scores = result.to_pandas()                             
    scores.insert(0, "id", df["id"].values)                
    scores["question_type"] = df["question_type"].values

    out = ROOT / "reports" / "ragas_scores.csv"             
    out.parent.mkdir(exist_ok=True)
    scores.to_csv(out, index=False)
    print(f"Saved {len(scores)} rows → {out}")

if __name__ == "__main__":
    main()
