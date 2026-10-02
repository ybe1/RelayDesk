import pytest
from types import SimpleNamespace


def test_round_robin_retains_later_query_results():
    from app.knowledge.multi_retrieval import merge_candidate_lists
    hits=merge_candidate_lists([[{'id':i} for i in range(60)],[{'id':0},{'id':90}],[{'id':91}]],limit=5)
    assert len(hits)==len({h['id'] for h in hits})==5
    assert {90,91}.issubset({h['id'] for h in hits})


async def test_one_embedding_batch_and_one_rerank_without_understanding():
    from app.knowledge.multi_retrieval import search_multi_queries
    from app.knowledge.retrieval import RetrievalDependencies
    calls=[]
    class Embed:
        def embed(self,texts): calls.append(('embed',texts)); return [[0.1]*1024 for _ in texts]
    class Store:
        def hybrid_search(self,vector,text,**kwargs):
            calls.append(('search',text,kwargs))
            return [{'id':1 if text=='原问题' else 2,'question':text,'answer':'政策答案'}]
    class Rerank:
        async def rerank(self,question,docs,top_n): calls.append(('rerank',question)); return [(i,0.9-i*0.1) for i in range(len(docs))]
    deps=RetrievalDependencies(None,Embed(),Store(),Rerank())
    result=await search_multi_queries('原问题',['原问题','保修'],deps)
    assert result.sufficient and len(result.hits)==2
    assert [c[0] for c in calls]==['embed','search','search','rerank']


async def test_incomplete_vector_batch_fails():
    from app.knowledge.multi_retrieval import search_multi_queries
    deps=SimpleNamespace(_closed=False,embedder=SimpleNamespace(embed=lambda texts:[]))
    with pytest.raises(ValueError): await search_multi_queries('问题',['问题'],deps)


async def test_invalid_category_never_embeds():
    from app.knowledge.multi_retrieval import search_multi_queries
    with pytest.raises(ValueError): await search_multi_queries('问题',['问题'],None,category='untrusted')


async def test_failed_recall_leg_cannot_return_partial_evidence():
    from app.knowledge.multi_retrieval import search_multi_queries
    from app.knowledge.retrieval import RetrievalDependencies
    class Store:
        def hybrid_search(self,vector,text,**kwargs):
            if text=='second': raise ConnectionError('offline')
            return [{'id':1,'question':'first','answer':'answer'}]
    deps=RetrievalDependencies(None,SimpleNamespace(embed=lambda texts:[[0.1]*1024 for _ in texts]),Store(),None)
    with pytest.raises(ConnectionError): await search_multi_queries('first',['first','second'],deps)
