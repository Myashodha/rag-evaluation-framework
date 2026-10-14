import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "reports"


def md(df):
    return df.to_markdown(index=False) if not df.empty else "_none_"


def main():
    parts = ["# RAG Evaluation Quality Report\n"]

    cls = pd.read_csv(R / "failure_classification.csv")
    col = "final_category" if "final_category" in cls.columns else "category"
    parts += ["## 1. Failure classification\n",
              md(cls[col].value_counts().rename_axis("category").reset_index(name="count")), "\n"]

    if (R / "ragas_scores.csv").exists():
        r = pd.read_csv(R / "ragas_scores.csv")
        cols = [c for c in ["llm_context_precision_with_reference", "context_recall",
                            "faithfulness", "answer_relevancy"] if c in r.columns]
        parts += ["## 2. Ragas averages\n", md(r[cols].mean().round(3).rename("mean").reset_index()),
                  f"\nNaN cells: {int(r[cols].isna().sum().sum())}\n"]

    if (R / "deepeval_scores.csv").exists():
        d = pd.read_csv(R / "deepeval_scores.csv")
        cols = [c for c in ["precision", "recall", "faithfulness", "relevancy"] if c in d.columns]
        parts += ["## 3. DeepEval averages\n", md(d[cols].mean().round(3).rename("mean").reset_index()),
                  f"\nRows scored: {len(d)}\n"]

    if (R / "scenario_results.csv").exists():
        s = pd.read_csv(R / "scenario_results.csv")
        parts += ["## 4. Deliberate failure scenarios\n",
                  md(s.groupby(["scenario", "diagnosis"]).size().reset_index(name="count")), "\n"]

    parts += ["## 5. Lowest scoring rows\n"]
    if (R / "ragas_scores.csv").exists() and "faithfulness" in r.columns:
        parts += [md(r.sort_values("faithfulness").head(3)[["id", "faithfulness"]]), "\n"]

    parts += ["## 6. Known limitations\n",
              "- Scenario `inj_link` not run (out of memory on the test machine).\n",
              "- Small local model used for generation in scenarios; judges used a stronger model.\n"]

    (R / "quality_report.md").write_text("\n".join(parts), encoding="utf-8")
    print("saved", R / "quality_report.md")


if __name__ == "__main__":
    main()