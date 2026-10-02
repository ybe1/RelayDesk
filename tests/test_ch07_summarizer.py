import asyncio
import json
import logging
import pytest
from langchain_core.messages import HumanMessage,AIMessage
from app.context.contracts import SummaryBatch,HistoryRecord,ContextConfig,LayerAnchors,SummarySegment

class MemoryRepo:
    def __init__(self): self.anchors=LayerAnchors(0,4); self.segments=[]
    def load(self,cid): return self.anchors,list(self.segments)
    def append(self,cid,expected,upto,content):
        if expected!=self.anchors.summary_upto: return None
        self.anchors=LayerAnchors(upto,self.anchors.layer1_from)
        if not content: return None
        row=SummarySegment(len(self.segments)+1,expected+1,upto,content)
        self.segments.append(row); return row

def batch():
    return SummaryBatch(1,0,4,(HistoryRecord(HumanMessage(content='订单1001物流一直未到，帮我核实。',id='u'),1,'t'),
        HistoryRecord(AIMessage(content='暂未核实。',id='a'),4,'t')),())

async def test_pending_summary_does_not_advance_boundary(caplog):
    from app.services.summarizer import SummaryWorker
    started,release=asyncio.Event(),asyncio.Event()
    class Model:
        async def ainvoke(self,messages,**kwargs):
            started.set(); await release.wait()
            return AIMessage(content=json.dumps({'summary':'用户询问订单1001物流迟迟未到，请求核实，目前问题尚未解决。'},ensure_ascii=False))
    repo=MemoryRepo()
    worker=SummaryWorker(repo,lambda:Model(),ContextConfig(),logging.getLogger('app.summary'))
    with caplog.at_level(logging.INFO):
        assert worker.submit(batch())
        await started.wait()
        assert not worker.submit(batch())
        assert repo.anchors.summary_upto==0
        release.set(); await worker.aclose()
    assert repo.anchors.summary_upto==4 and len(repo.segments)==1
    assert 'summary start' in caplog.text and 'summary done' in caplog.text

async def test_failure_does_not_advance_and_can_retry():
    from app.services.summarizer import SummaryWorker
    class Model:
        async def ainvoke(self,*args,**kwargs): raise ConnectionError('offline')
    repo=MemoryRepo(); worker=SummaryWorker(repo,lambda:Model(),ContextConfig(),logging.getLogger('app.summary'))
    worker.submit(batch()); await worker.aclose()
    assert repo.anchors.summary_upto==0 and not repo.segments

def test_new_identifier_or_old_background_is_rejected():
    from app.services.summarizer import validate_summary
    with pytest.raises(ValueError): validate_summary('用户报告订单9999仍未签收，希望核实物流。',batch())

def test_no_facts_is_allowed():
    from app.services.summarizer import validate_summary
    greeting=SummaryBatch(1,0,2,(HistoryRecord(HumanMessage(content='你好',id='u'),1,'t'),),())
    assert validate_summary('',greeting)==''
