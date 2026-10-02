import asyncio
import json
import logging
from uuid import uuid4
from langchain_core.messages import HumanMessage,AIMessage
from app.context.contracts import ContextConfig
from app.schemas.chat import ChatRequest
from app.db.session import SessionLocal
from app.db.models import ConversationSummary
from tests.ch06_service_fixture import remove_case

async def test_smalltalk_logs_history_without_model(tmp_path,caplog):
    from app.workflow.runtime import checkpoint_runtime
    from app.workflow.graph import build_workflow
    from app.workflow.service import WorkflowChatService
    from app.services.chat import DatabaseChatStore
    from tests.test_ch05_agent_graph import GraphModel
    cid=None
    async with checkpoint_runtime(tmp_path/'history.sqlite') as saver:
        model=GraphModel([])
        service=WorkflowChatService(build_workflow(checkpointer=saver),DatabaseChatStore(SessionLocal),model_factory=lambda:model)
        try:
            with caplog.at_level(logging.INFO,logger='app.context'):
                events=[e async for e in service.events(ChatRequest(user_id='ch07-'+uuid4().hex,message='你好'))]
            cid=events[0][1]['conversation_id']
            assert events[-1][0]=='done' and model.calls==0
            assert 'history_ctx' in caplog.text
            snap=await service.graph.aget_state(service._config(cid))
            assert len(snap.values['messages'])==2
            assert len(snap.values['message_db_ids'])==2
        finally:
            await service.summary_worker.aclose()
            if cid: remove_case(cid)

async def test_blocked_summary_does_not_block_current_sse(tmp_path,caplog):
    from app.workflow.runtime import checkpoint_runtime
    from app.workflow.graph import build_workflow
    from app.workflow.service import WorkflowChatService
    from app.services.chat import DatabaseChatStore
    from app.services.summarizer import SummaryWorker
    from app.repositories.summaries import SummaryRepository
    from tests.test_ch05_agent_graph import GraphModel
    from sqlalchemy import delete
    started,release=asyncio.Event(),asyncio.Event()
    class Blocked:
        async def ainvoke(self,messages,**kwargs):
            started.set(); await release.wait()
            return AIMessage(content=json.dumps({'summary':'用户报告演示订单1001物流未到并请求核实，目前该问题尚未解决。'},ensure_ascii=False))
    cfg=ContextConfig(window=18000,output=2000,user_input=2000,steps=3,tool_result=1200,rerank_k=5)
    repo=SummaryRepository(SessionLocal)
    worker=SummaryWorker(repo,lambda:Blocked(),cfg,logging.getLogger('app.summary'))
    user='ch07-block-'+uuid4().hex
    store=DatabaseChatStore(SessionLocal); cid=store.start_or_load(user,None)
    async with checkpoint_runtime(tmp_path/'blocked.sqlite') as saver:
        graph=build_workflow(checkpointer=saver)
        service=WorkflowChatService(graph,store,model_factory=lambda:GraphModel([]),context_config=cfg,
            summary_repository=repo,summary_worker=worker)
        try:
            messages=[]; ids={}
            for n in range(12):
                for m in [HumanMessage(content='订单1001物流未到，请核实。'+'补充描述。'*150,id=f'old{n}:user'),
                          AIMessage(content='待核实。'*150,id=f'old{n}:final')]:
                    ids[m.id]=service._save(cid,f'old{n}',m); messages.append(m)
            await graph.aupdate_state(service._config(cid),{'messages':messages,'message_db_ids':ids},as_node='log')
            async def consume(): return [e async for e in service.events(ChatRequest(user_id=user,conversation_id=cid,message='你好'))]
            task=asyncio.create_task(consume())
            await asyncio.wait_for(started.wait(),3)
            events=await asyncio.wait_for(task,3)
            assert events[-1][0]=='done' and not release.is_set()
            assert repo.load(cid)[0].summary_upto==0
            release.set(); await worker.aclose()
            assert repo.load(cid)[0].summary_upto>0
            assert repo.load(cid)[1]
        finally:
            release.set(); await worker.aclose()
            with SessionLocal() as session:
                session.execute(delete(ConversationSummary).where(ConversationSummary.conversation_id==cid));session.commit()
            remove_case(cid)
