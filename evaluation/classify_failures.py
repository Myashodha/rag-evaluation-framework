import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.dataset import load_golden, DOCS_DIR
from src.scenarios import normalize, unsupported_numbers, is_refusal

THRESH = 0.7


def key_facts(text: str) -> list[str]:
    """Numbers and number+unit phrases from a ground truth, e.g. '30 days', '1000 inr'."""
    t = normalize(text)
    return re.findall(r"\d[\d,]*(?:\s?(?:business days|days|hours|inr|%))?", t)


def doc_text(sources: list[str]) -> str:
    return normalize(" ".join((DOCS_DIR / s).read_text(encoding="utf-8") for s in sources))


def classify(row, gt_row, ragas):
    contexts = json.loads(row["contexts"])
    ctx = normalize(" ".join(contexts))
    answer = normalize(row["answer"])
    facts = [f.strip() for f in key_facts(gt_row.ground_truth)]
    docs = doc_text(gt_row.expected_sources)

    if is_refusal(row["answer"]):
        return "OVER_REFUSAL", "answerable question was refused"

    # facts in ground truth that appear in neither the document nor the context
    if facts and not all(f in docs for f in facts):
        return "DATASET_ERROR", f"ground truth facts not in document: {[f for f in facts if f not in docs]}"

    if facts and not all(f in ctx for f in facts):
        return "RETRIEVAL_FAILURE", f"missing from context: {[f for f in facts if f not in ctx]}"

    bad = unsupported_numbers(row["answer"], contexts)
    if bad:
        return "HALLUCINATION", f"numbers not in context: {bad}"

    if facts and not all(f in answer for f in facts):
        return "INCOMPLETE_ANSWER", f"answer omits: {[f for f in facts if f not in answer]}"

    if ragas is not None:
        low = [m for m in ("faithfulness", "context_recall", "llm_context_precision_with_reference",
                           "answer_relevancy") if pd.notna(ragas.get(m)) and ragas[m] < THRESH]
        if low:
            return "REVIEW_MANUALLY", f"deterministic checks passed but judge scored low on {low}"

    return "OK", ""


def main():
    outputs = pd.read_csv(ROOT / "reports" / "rag_outputs.csv")
    gold = {r.id: r for r in load_golden()}
    scores_path = ROOT / "reports" / "ragas_scores.csv"
    ragas = pd.read_csv(scores_path).set_index("id").to_dict("index") if scores_path.exists() else {}

    rows = []
    for _, row in outputs.iterrows():
        cat, why = classify(row, gold[row["id"]], ragas.get(row["id"]))
        rows.append({"id": row["id"], "question": row["question"], "category": cat, "reason": why,
                     "answer": row["answer"][:150].replace("\n", " ")})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "failure_classification.csv", index=False)

    print(df["category"].value_counts())
    print("\nNeeds attention:")
    print(df[df["category"] != "OK"][["id", "category", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()