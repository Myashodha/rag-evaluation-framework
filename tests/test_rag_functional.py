import re
import pytest   


from src.dataset import load_golden, DOCS_DIR
from src.rag import NO_ANSWER

ROWS=load_golden()
ANSWERABLE=[r for r in ROWS if r.answerable ]
UNANSWERABLE=[r for r in ROWS if not r.answerable]
ALLOWED_DOCS={f.name for f in DOCS_DIR.glob("*.txt")}

SENSITIVE_PATTERNS=[
    r"gsk[A-Za-z0-9]+",
    r"\b\d{3}-\d{2}-\d{4}\b",          # SSN-like
    r"\b(?:\d[ -]*?){13,16}\b",        # card-like number
]

CLARIFY_MARKERS=["clarify","do you mean","could you","?"]

@pytest.mark.parametrize("row",ANSWERABLE,ids=lambda r:r.id)
def test_plumbing(rag,row):
    out=rag.ask(row.question)

    assert out is not None
    assert out["answer" ] ,"empty answer"
    assert len(out["contexts"])>0,"nothing retrieved"
    assert set(out["sources"]) <= ALLOWED_DOCS,"source is not in knowlede base"

@pytest.mark.parametrize("row",ANSWERABLE,ids=lambda r :r.id)
def test_expected_source_retrieved(rag,row):
    out=rag.ask(row.question)

    assert set(row.expected_sources) & set(out["sources"]),(
      f"{row.id}:expected {row.expected_sources} ,got{out['sources']}")


@pytest.mark.parametrize("row",ANSWERABLE,ids=lambda r :r.id)
def test_no_sensitive_data(rag,row):
    answer=rag.ask(row.question)["answer"]
    for patterns in SENSITIVE_PATTERNS:
        assert not re.search(patterns,answer),f"{row.id} matched {patterns}"


@pytest.mark.parametrize("row",[r for r in UNANSWERABLE if r.question_type == "out_of_scope"],ids=lambda r : r.id)
def test_out_of_scope_refuse(rag,row):
    answer=rag.ask(row.question)["answer"]
    assert NO_ANSWER.lower() in answer.lower(),f"{row.id} did not refuse {answer}"

@pytest.mark.parametrize("row",[r for r in UNANSWERABLE if r.question_type == "ambiguous"],ids=lambda r :r.id)
def test_ambiguious_refuses_or_clarifies(rag,row):
    answer=rag.ask(row.question)["answer"]
    refused =NO_ANSWER.lower() in answer.lower()
    clarified=any( r in answer.lower()  for r in CLARIFY_MARKERS)
    assert refused or clarified, f"{row.id} invented an answer: {answer}"

