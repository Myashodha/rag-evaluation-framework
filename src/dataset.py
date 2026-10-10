from pathlib import Path

import pandas as pd
from pydantic import BaseModel, field_validator

ROOT=Path(__file__).resolve().parent.parent
DATASET_PATH=ROOT /"evaluation"/"rag_dataset.csv"
DOCS_DIR=ROOT/ "data" / "documents"


class GoldenRow(BaseModel):
    id:str
    question:str
    ground_truth:str
    expected_source:str =""
    category:str
    question_type:str
    answerable:bool

    @field_validator("answerable",mode='before')
    @classmethod
    def parse_bool(cls,v):
        return str(v).strip().lower() == "true"

    @property
    def expected_sources(self)  -> list[str]:
        return [s.strip() for s in self.expected_source.split("|") if s.strip()]    

def load_golden(path=DATASET_PATH)->list[GoldenRow]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return[GoldenRow(**row) for row in df.to_dict(orient="records")]