import json,sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import load_golden
from src.rag import RAGPipeline

OUT=Path(__file__).parent.parent / "reports" / "rag_outputs.csv"

def main():
    rag=RAGPipeline()
    rag.build_index()

    records=[]
    for r in load_golden():
        out=rag.ask(r.question)
        records.append({
            "id":r.id,
            "question":r.question,
            "answer":out["answer"],
            "contexts":json.dumps(out["contexts"]),
            "sources": json.dumps(out["sources"]),
            "ground_truth":r.ground_truth,
            "expected_source":r.expected_source,
            "question_type":r.question_type
            
        })

    OUT.parent.mkdir(exist_ok="True")
    pd.DataFrame(records).to_csv(OUT,index=False)
    print("saved", OUT, len(records), "rows")


if __name__ == "__main__":
    main()
    

