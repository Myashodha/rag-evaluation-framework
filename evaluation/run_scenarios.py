import sys
import tempfile
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.rag import RAGPipeline
from src.scenarios import (
    FACT_CASES, OUT_OF_SCOPE_CASES, INJECTIONS, WrongAnswerLLM, HardenedRAG,
    diagnose, make_injected_dir, attack_succeeded,
)

records = []


def log(scenario, case_id, question, diagnosis, out):
    records.append({"scenario": scenario, "id": case_id, "question": question,
                    "diagnosis": diagnosis, "sources": out["sources"],
                    "answer": out["answer"][:200].replace("\n", " ")})
    print(f"{scenario:22} {case_id:11} {diagnosis}")


# ---- baseline (healthy system) ----
healthy = RAGPipeline(collection_name="scn_healthy",embedder=EMB)
healthy.build_index()
for c in FACT_CASES:
    out = healthy.ask(c["q"])
    log("0_baseline", c["id"], c["q"], diagnose(out, c["facts"]), out)

# ---- Scenario 1: wrong chunk (tiny chunks, top_k=1) ----
broken = RAGPipeline(chunk_size=60, chunk_overlap=0, top_k=1, collection_name="scn_broken",embedder=EMB)
broken.build_index()
for c in FACT_CASES:
    out = broken.ask(c["q"])
    log("1_wrong_chunk", c["id"], c["q"], diagnose(out, c["facts"]), out)

# ---- Scenario 2: correct context, wrong answer ----
liar = RAGPipeline(llm=WrongAnswerLLM(), collection_name="scn_liar",embedder=EMB)
liar.build_index()
for c in FACT_CASES:
    out = liar.ask(c["q"])
    log("2_bad_generation", c["id"], c["q"], diagnose(out, c["facts"]), out)

# ---- Scenario 3: out of scope ----
for c in OUT_OF_SCOPE_CASES:
    out = healthy.ask(c["q"])
    log(f"3_out_of_scope_{c['kind']}", c["id"], c["q"],
        diagnose(out, None, answerable=False), out)

# ---- Scenario 4: prompt injection (baseline vs hardened) ----
for name, spec in INJECTIONS.items():
    d = make_injected_dir(Path(tempfile.mkdtemp()) / name, name)
    for label, cls in [("baseline", RAGPipeline), ("hardened", HardenedRAG)]:
        p = cls(docs_dir=d, collection_name=f"scn_{name}_{label}",embedder=EMB)
        p.build_index()
        out = p.ask(spec["question"])
        if not any(name in s for s in out["sources"]):
            verdict = "INJECTION_NOT_RETRIEVED"
        else:
            verdict = "ATTACK_SUCCEEDED" if attack_succeeded(out["answer"], name) else "RESISTED"
        log(f"4_injection_{label}", name, spec["question"], verdict, out)

df = pd.DataFrame(records)
(ROOT / "reports").mkdir(exist_ok=True)
df.to_csv(ROOT / "reports" / "scenario_results.csv", index=False)
print("\nSummary:")
print(df.groupby(["scenario", "diagnosis"]).size())