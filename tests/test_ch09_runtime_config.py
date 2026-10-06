import hashlib


def test_runtime_confidence_loads_only_matching_active_configuration(tmp_path):
    from app.evaluation.confidence_calibration import ScoredCase, save_report, select_threshold
    from app.evaluation.runtime_config import corpus_hash, load_runtime_confidence
    kb=tmp_path/"kb"; kb.mkdir(); (kb/"b.md").write_text("乙",encoding="utf-8"); (kb/"a.md").write_text("甲",encoding="utf-8")
    expected=hashlib.sha256("甲乙".encode()).hexdigest()
    assert corpus_hash(kb)==expected
    report=select_threshold([
        ScoredCase(case_id="yes",fact_group="a",expected_allowed=True,score=.9),
        ScoredCase(case_id="no",fact_group="b",expected_allowed=False,score=.1),
    ],candidate_thresholds=[.5],metadata={"dataset_hash":"d","corpus_hash":expected,"rerank_model":"r","top_k":5})
    path=tmp_path/"report.json"; save_report(report,path)
    assert load_runtime_confidence(path,kb,rerank_model="r",top_k=5).threshold==.5
