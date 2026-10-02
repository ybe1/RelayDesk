from types import SimpleNamespace
from app.workflow.budget import TurnBudget
from app.context.contracts import ContextConfig

async def test_demo_rerank_top_k_follows_turn_config(monkeypatch):
    from app.workflow.retrieval import retrieve_for_turn
    from app.knowledge import retrieval
    recorded=[]
    class Owned:
        model=object()
        async def aclose(self): pass
    async def dependencies(**kwargs): return Owned()
    async def search(query,**kwargs): recorded.append(kwargs['top_k']);return object()
    monkeypatch.setattr(retrieval.RetrievalDependencies,'afrom_settings',dependencies)
    monkeypatch.setattr(retrieval,'search_knowledge_ch04',search)
    budget=TurnBudget();budget.context_config=ContextConfig(rerank_k=5)
    await retrieve_for_turn('退货政策',budget)
    assert recorded==[5]
