import json

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import Conversation, LowConfidenceQuestion, ReviewQueue
from app.flywheel.processor import process_pending


class Model:
    def __init__(self, outputs): self.outputs, self.calls, self.payloads = iter(outputs), 0, []
    def with_structured_output(self, _schema, **_kwargs): return self
    async def ainvoke(self, messages):
        self.calls += 1
        self.payloads.append(messages[-1].content)
        return next(self.outputs)


def harness(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'flywheel.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE conversations (id INTEGER PRIMARY KEY, user_id VARCHAR(64), status VARCHAR(16), summary TEXT, summary_upto_msg_id INTEGER, layer1_from_msg_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        connection.exec_driver_sql("CREATE TABLE review_queue (id INTEGER PRIMARY KEY AUTOINCREMENT, normalized_question VARCHAR(512) NOT NULL, ai_suggested_answer TEXT, occurrence_count INTEGER NOT NULL DEFAULT 1, review_status VARCHAR(16) NOT NULL DEFAULT '待审', approved_answer TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        connection.exec_driver_sql("CREATE TABLE low_confidence_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER, raw_question TEXT NOT NULL, source VARCHAR(32) NOT NULL, reason TEXT, retrieved_chunks JSON, matched_review_id INTEGER, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        session.add(Conversation(id=1, user_id="u", status="进行中")); session.commit()
    return factory


def add_gap(factory, text="邮费多少", reason="证据不足", chunks=None):
    with factory() as session:
        row = LowConfidenceQuestion(conversation_id=1, raw_question=text,
            source="retrieval_low_conf", reason=reason, retrieved_chunks=chunks)
        session.add(row); session.commit(); return row.id


@pytest.mark.parametrize("status", ["待审", "通过", "驳回"])
async def test_match_increments_without_changing_any_status(status, tmp_path):
    factory = harness(tmp_path)
    with factory() as session:
        existing = ReviewQueue(normalized_question="配送费用是多少", ai_suggested_answer="满99包邮",
            review_status=status, approved_answer="核准答案" if status == "通过" else None)
        session.add(existing); session.commit(); existing_id = existing.id
    gap_id = add_gap(factory)
    model = Model([{"normalized_question":"邮费是多少","ai_suggested_answer":"参考运费规则",
                    "matched_question_id":existing_id}])
    report = await process_pending(session_factory=factory, model=model,
        state_path=tmp_path / "state.json")
    with factory() as session:
        refreshed = session.get(ReviewQueue, existing_id)
        assert refreshed.occurrence_count == 2
        assert refreshed.review_status == status
        assert refreshed.approved_answer == ("核准答案" if status == "通过" else None)
        assert session.get(LowConfidenceQuestion, gap_id).matched_review_id == existing_id
    assert report.merged == report.model_calls == model.calls == 1


async def test_new_gap_creates_review_and_deidentifies_phone(tmp_path):
    factory = harness(tmp_path); add_gap(factory, "我手机号13800000000，邮费怎么算")
    model = Model([{"normalized_question":"配送费用如何计算？","ai_suggested_answer":"参考配送规则",
                    "matched_question_id":None}])
    report = await process_pending(session_factory=factory, model=model, state_path=tmp_path / "state.json")
    assert report.created == report.model_calls == 1
    assert "13800000000" not in model.payloads[0] and "[手机号]" in model.payloads[0]
    with factory() as session:
        review = session.scalar(select(ReviewQueue))
        assert review.normalized_question == "配送费用如何计算？"


async def test_unknown_candidate_id_aborts_without_linking(tmp_path):
    factory = harness(tmp_path); gap_id = add_gap(factory)
    model = Model([{"normalized_question":"配送费用如何计算？","ai_suggested_answer":None,
                    "matched_question_id":999}])
    with pytest.raises(ValueError, match="not supplied"):
        await process_pending(session_factory=factory, model=model, state_path=tmp_path / "state.json")
    with factory() as session:
        assert session.get(LowConfidenceQuestion, gap_id).matched_review_id is None


async def test_operational_failure_is_skipped_before_model_call(tmp_path):
    factory = harness(tmp_path); add_gap(factory, reason="retrieval_unavailable:TimeoutError")
    model = Model([]); state = tmp_path / "state.json"
    report = await process_pending(session_factory=factory, model=model, state_path=state)
    assert report.skipped == 1 and model.calls == 0
    assert json.loads(state.read_text(encoding="utf-8"))["operational_skips"]


@pytest.mark.parametrize("reason", ["order_unavailable", "expansion_unavailable:TimeoutError"])
async def test_after_sales_operational_failure_is_skipped(reason, tmp_path):
    factory=harness(tmp_path); add_gap(factory, reason=reason)
    report=await process_pending(session_factory=factory, model=Model([]), state_path=tmp_path/"state.json")
    assert report.skipped == 1


async def test_skipped_and_failed_rows_do_not_starve_later_rows(tmp_path):
    factory=harness(tmp_path)
    skipped=add_gap(factory, reason="order_unavailable")
    failed=add_gap(factory, text="第一条坏数据")
    add_gap(factory, text="订单A1001邮费", chunks={"query":"标准邮费问题","hits":[]})
    class SometimesBad(Model):
        async def ainvoke(self,messages):
            self.calls+=1; self.payloads.append(messages[-1].content)
            if self.calls == 1: raise RuntimeError("bad model row")
            return {"normalized_question":"配送费用是多少？","ai_suggested_answer":"参考规则","matched_question_id":None}
    state=tmp_path/"state.json"
    first=await process_pending(session_factory=factory,model=SometimesBad([]),batch_size=3,state_path=state)
    assert first.skipped == 2 and first.created == 1
    saved=json.loads(state.read_text(encoding="utf-8"))
    assert skipped in saved["operational_skips"] and failed in saved["failed"]
    model=Model([])
    again=await process_pending(session_factory=factory,model=model,batch_size=3,state_path=state)
    assert again.scanned == 0 and model.calls == 0


async def test_uses_resolved_snapshot_query_and_redacts_order_id(tmp_path):
    factory=harness(tmp_path); add_gap(factory,text="帮我看订单100123456789",chunks={"query":"订单100123456789的邮费是多少","hits":[]})
    model=Model([{"normalized_question":"订单的配送费用是多少？","ai_suggested_answer":None,"matched_question_id":None}])
    await process_pending(session_factory=factory,model=model,state_path=tmp_path/"state.json")
    assert "100123456789" not in model.payloads[0]
    assert "[订单号]的邮费是多少" in model.payloads[0]


def test_labeled_cases_cover_required_boundaries():
    rows = json.loads(open("tests/data/ch09_flywheel_cases.json", encoding="utf-8").read())
    assert {row["id"] for row in rows} == {"paraphrase","model","negation","reference","passed","rejected","operational"}
