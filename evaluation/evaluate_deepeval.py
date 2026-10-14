import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.dataset import load_golden
from src.deepeval_judge import JudgeLLM

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20     # DeepEval on a subset


def build_metrics(judge):
    kw = dict(model=judge, include_reason=True, async_mode=False)   # sequential = gentler on Groq
    return {
        "precision": ContextualPrecisionMetric(threshold=0.7, **kw),
        "recall": ContextualRecallMetric(threshold=0.7, **kw),
        "faithfulness": FaithfulnessMetric(threshold=0.8, **kw),
        "relevancy": AnswerRelevancyMetric(threshold=0.7, **kw),
    }


def main():
    df = pd.read_csv(ROOT / "reports" / "rag_outputs.csv").head(LIMIT)
    truth = {r.id: r.ground_truth for r in load_golden()}     # answer key from the dataset
    judge = JudgeLLM("groq")

    rows = []
    for r in df.itertuples():
        case = LLMTestCase(
            input=r.question,
            actual_output=r.answer,
            expected_output=truth[r.id],
            retrieval_context=json.loads(r.contexts),
        )
        row = {"id": r.id}
        for name, metric in build_metrics(judge).items():     # fresh metric per case
            try:
                metric.measure(case)
                row[name] = metric.score
                row[f"{name}_reason"] = metric.reason
            except Exception as e:
                row[name], row[f"{name}_reason"] = None, f"ERROR: {str(e)[:150]}"
        rows.append(row)
        print(r.id, {k: v for k, v in row.items() if not k.endswith("reason")})

    out = ROOT / "reports" / "deepeval_scores.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()