import pytest
from app.knowledge.retrieval import RetrievalResult
from app.knowledge.query_understanding import QueryUnderstanding


def result(score=0.8, sufficient=True):
    return RetrievalResult(QueryUnderstanding(normalized_query="运费", search_query="运费"),
        [{"id": 1, "rerank_score": score, "question": "运费", "answer": "8元", "section_path": "政策/运费"}], sufficient, "ok")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), None])
def test_invalid_score_is_rejected(score):
    from app.workflow.retrieval import gate_result
    assert not gate_result(result(score), 0.5).allowed


def test_gate_boundary_empty_and_mechanical_refusal():
    from app.workflow.retrieval import gate_result
    assert gate_result(result(0.5), 0.5).allowed
    assert not gate_result(result(0.49), 0.5).allowed
    assert not gate_result(result(0.8, False), 0.5).allowed
    empty = result()
    empty.hits = []
    assert not gate_result(empty, 0.5).allowed


def test_citations_are_unique_across_multiple_retrievals():
    from app.workflow.retrieval import CitationRegistry
    registry = CitationRegistry()
    first = registry.merge(result().hits)
    second = registry.merge(result().hits + [{"id": 2, "question": "退货", "answer": "7天", "section_path": "政策/退货"}])
    assert [r["n"] for r in first] == [1]
    assert [r["n"] for r in second] == [1, 2]
    assert len(registry.citations) == 2


async def test_retrieval_budget_shared_and_dependencies_closed_on_failure(monkeypatch):
    from app.workflow.retrieval import retrieve_for_turn
    from app.workflow.budget import TurnBudget, Limits, BudgetExceeded
    from app.knowledge.retrieval import RetrievalDependencies
    class Owned:
        model = object()
        closed = False
        async def aclose(self): self.closed = True
    owned = Owned()
    async def factory(): return owned
    monkeypatch.setattr(RetrievalDependencies, "afrom_settings", factory)
    async def search(query, *, dependencies):
        assert dependencies.model.budget is budget
        dependencies.model.budget.reserve_llm(1, 1)
    monkeypatch.setattr("app.knowledge.retrieval.search_knowledge_ch04", search)
    budget = TurnBudget(Limits(model_calls=0))
    with pytest.raises(BudgetExceeded): await retrieve_for_turn("用户原话", budget)
    assert owned.closed


async def test_structured_model_is_budgeted():
    from app.workflow.model import BudgetedModel
    from app.workflow.budget import TurnBudget
    from langchain_core.messages import HumanMessage
    class Raw:
        def with_structured_output(self, schema, **kwargs): return self
        async def ainvoke(self, messages, **kwargs): return {"value": 1}
    budget = TurnBudget()
    result = await BudgetedModel(Raw(), budget).with_structured_output(dict).ainvoke([HumanMessage(content="x")])
    assert result == {"value": 1}
    assert budget.snapshot()["model_calls"] == 1
    assert budget.snapshot()["estimated"]
