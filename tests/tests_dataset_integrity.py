from src.dataset import load_golden,DOCS_DIR

ROW=load_golden()

def ids_are_unique() :
   ids =[ r.id for r in ROW]
   assert len(ids) == len(set(ids))

def test_no_empty_fields():
    for r in ROW:
        assert r.question.strip(),r.id
        assert r.ground_truth.strip(),r.id

def test_expected_source_exists():
    for r in ROW:
        for src in r.expected_sources:
            assert (DOCS_DIR /src ).exists(),f"{r.id}:{src} not found"

def test_answerble_rows_have_sources():
    for r in ROW:
        if r.answerable:
           assert r.expected_sources ,f"{r.id} has no expected source"

def test_unanswerable_rows_donot_have_source():
    for r in ROW:
        if not r.answerable:
            assert not r.expected_sources,r.id