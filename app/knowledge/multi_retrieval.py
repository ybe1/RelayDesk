"""Fair multi-query hybrid recall, followed by one aggregate rerank."""
import asyncio
from app.config import settings
from .query_understanding import QueryUnderstanding
from .retrieval import RetrievalResult
from .hybrid_store import build_filter
from .embedding import validate_vectors
from .rerank import MAX_DOCUMENT_BYTES,MAX_TOTAL_DOCUMENT_BYTES


def merge_candidate_lists(lists,limit=50):
    iterators=[iter(items) for items in lists]
    seen=set()
    result=[]
    while iterators and len(result)<limit:
        active=[]
        for iterator in iterators:
            for item in iterator:
                if item['id'] in seen: continue
                seen.add(item['id']); result.append(item); active.append(iterator)
                break
            if len(result)==limit: return result
        iterators=active
    return result


async def search_multi_queries(question,queries,dependencies,category=None,*,top_k=10):
    if not 1<=len(queries)<=3 or any(not isinstance(q,str) or not q.strip()
        or len(q)>(4096 if i==0 and q==question else 512) for i,q in enumerate(queries)):
        raise ValueError('invalid expanded queries')
    if category is not None and category not in settings.rag_category_allowlist: raise ValueError('invalid category')
    if dependencies._closed: raise RuntimeError('retrieval dependencies are closed')
    understood=QueryUnderstanding(normalized_query=question,search_query=question,category=category)
    vectors=await asyncio.to_thread(dependencies.embedder.embed,queries)
    validate_vectors(vectors,len(queries))
    lists=[]
    for vector,query in zip(vectors,queries):
        lists.append(await asyncio.to_thread(dependencies.store.hybrid_search,vector,query,
            recall_limit=50,output_limit=50,filter_expr=build_filter(category=category)))
    candidates=merge_candidate_lists(lists)
    if not candidates: return RetrievalResult(understood,[],False,'no_hits')
    cap=min(MAX_DOCUMENT_BYTES,MAX_TOTAL_DOCUMENT_BYTES//len(candidates))
    docs=[f"问题：{h['question']}\n答案：{h['answer']}".encode('utf-8')[:cap].decode('utf-8',errors='ignore') for h in candidates]
    scores=await dependencies.reranker.rerank(question,docs,top_n=top_k)
    if not scores: return RetrievalResult(understood,[],False,'no_rerank_hits')
    hits=[{**candidates[i],'rerank_score':score,'rank':rank} for rank,(i,score) in enumerate(sorted(scores,key=lambda item:item[1],reverse=True),1)]
    sufficient=hits[0]['rerank_score']>=dependencies.rerank_min_score
    return RetrievalResult(understood,hits[::2]+hits[1::2][::-1],sufficient,'retrieval_ok' if sufficient else 'rerank_below_threshold')
