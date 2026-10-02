import json
import pytest
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage


@pytest.mark.parametrize('node',['resolve','intent'])
async def test_understanding_does_not_serialize_ui_citation_metadata(node):
    from app.workflow.resolution import resolve_question
    from app.workflow.intent import classify_intent
    from app.workflow.budget import TurnBudget,Limits
    class Model:
        async def ainvoke(self,messages,**kwargs):
            self.sent=messages
            return AIMessage(content=json.dumps({'intent':'订单','confidence':1} if node=='intent' else
                {'question':'订单1001怎么样','reference_status':'explicit','scope':'order_specific','order_id':'1001','order_source':'current'}))
    model=Model()
    history=[HumanMessage(content='订单1001怎么样',id='u'),AIMessage(content='还在配送',id='a',
        additional_kwargs={'citations':[{'answer':'不应发送的引用原文'*5000}]})]
    if node=='intent': await classify_intent(model,'订单1001怎么样',history)
    else: await resolve_question(model,'订单1001怎么样',history,TurnBudget(Limits(tokens=1000000)),shared_history=True)
    assert '不应发送的引用原文' not in model.sent[-1].content
    assert len(model.sent[-1].content)<2000
    assert history[1].additional_kwargs['citations']


def test_recovered_tool_result_precedes_later_turn_in_projection():
    from app.workflow.history import merge_history,close_missing_results
    from app.context.layers import history_records
    first=HumanMessage(content='查订单1001',id='u1')
    call=AIMessage(content='',id='a1',tool_calls=[{'id':'c','name':'query_order','args':{}}])
    next_turn=HumanMessage(content='你好',id='u2')
    history=merge_history([],[first,call,next_turn])
    history.extend(close_missing_results(history))
    records=history_records(history,{'u1':101,'a1':102,'u2':103})
    assert [r.message.type for r in records]==['human','ai','tool','human']
    assert records[2].turn_id=='u1'


def test_summary_does_not_cover_discarded_tool_facts():
    import logging
    from app.services.summarizer import SummaryWorker
    from app.context.contracts import SummaryBatch,HistoryRecord,ContextConfig
    from tests.test_ch07_summarizer import MemoryRepo
    cfg=ContextConfig(window=18000,output=2000,user_input=2000,steps=3,tool_result=1200,rerank_k=5)
    tool=ToolMessage(content=json.dumps({'product_id':'P987','details':'重要条件'*6000}),tool_call_id='c')
    batch=SummaryBatch(1,0,4,(HistoryRecord(HumanMessage(content='查询商品'),1,'t'),HistoryRecord(tool,4,'t')),())
    worker=SummaryWorker(MemoryRepo(),lambda:None,cfg,logging.getLogger('test'))
    with pytest.raises(ValueError,match='summary turn exceeds window'): worker._parts(batch)


def test_agent_log_uses_actual_growing_summary_budget(monkeypatch):
    from types import SimpleNamespace
    import app.workflow.agent as agent
    from app.context.contracts import ContextConfig,HistoryProjection,LayerAnchors
    from app.workflow.budget import TurnBudget
    captured=[]
    monkeypatch.setattr(agent,'record_model',lambda *args:captured.append(args[-1]))
    model=SimpleNamespace(bind_tools=lambda *args,**kwargs:SimpleNamespace(kwargs={}))
    cfg=ContextConfig()
    context=SimpleNamespace(context_config=cfg,budget=TurnBudget(),history_projection=HistoryProjection(LayerAnchors(),(),(),'摘'*1200,0,()))
    wrapper=agent._model(SimpleNamespace(model=model),context,{'turn_id':'t','conversation_id':1},'decision',[])
    wrapper.context_logger([],[])
    assert captured[0].summary==1200


async def test_capacity_error_is_actionable_in_sse():
    from app.api.chat import stream_chat
    from app.context.budget import ContextCapacityError
    from app.schemas.chat import ChatRequest
    class Service:
        async def events(self,req):
            raise ContextCapacityError('用户输入超出上下文额度，请缩短后重试')
            yield
    response=stream_chat(ChatRequest(user_id='u',message='过长'),Service(),None)
    content=''.join([part async for part in response.body_iterator])
    assert '请缩短后重试' in content and '服务暂时不可用' not in content


def test_dialogue_false_promise_does_not_pass_live_quality():
    from scripts.eval_ch07 import assess_dialogue
    report={'mode':'live','turns':[{'passed':True,'anchors':{'layer1_from':0},'answer':'订单1001已核实，手机号13800000000，保证明天到货。'} for _ in range(22)]}
    result=assess_dialogue(report,'default')
    assert not result['passed']
    assert result['live_quality']!='passed'


async def test_cancelled_persisted_request_recovers_in_real_checkpoint(tmp_path):
    import asyncio
    from uuid import uuid4
    from langgraph.graph import StateGraph,START,END
    from app.workflow.runtime import checkpoint_runtime
    from app.workflow.contracts import WorkflowState,TurnContext
    from app.workflow.service import WorkflowChatService
    from app.services.chat import DatabaseChatStore
    from app.db.session import SessionLocal
    from app.schemas.chat import ChatRequest
    from tests.ch06_service_fixture import remove_case
    entered=asyncio.Event()
    calls=0
    async def log(state,runtime):
        nonlocal calls
        if calls: return {}
        calls+=1
        request=AIMessage(content='',id=state['turn_id']+':call',tool_calls=[{'id':'cancelled','name':'query_order','args':{'order_id':'1001'}}])
        await runtime.context.save_message(request)
        entered.set()
        await asyncio.Event().wait()
    builder=StateGraph(WorkflowState,context_schema=TurnContext)
    builder.add_node('log',log);builder.add_edge(START,'log');builder.add_edge('log',END)
    user='ch07-cancel-'+uuid4().hex
    async with checkpoint_runtime(tmp_path/'cancel.sqlite') as saver:
        service=WorkflowChatService(builder.compile(checkpointer=saver),DatabaseChatStore(SessionLocal),model_factory=lambda:None)
        prepared=await service.prepare(ChatRequest(user_id=user,message='订单1001怎么样'))
        cid=prepared.conversation_id
        async def consume(): return [e async for e in service.events(ChatRequest(user_id=user,conversation_id=cid,message='订单1001怎么样'),prepared)]
        task=asyncio.create_task(consume())
        try:
            await asyncio.wait_for(entered.wait(),3);task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            snapshot=await service.graph.aget_state(service._config(cid))
            assert [m.type for m in snapshot.values['messages']]==['human','ai','tool']
            assert 'interrupted' in snapshot.values['messages'][-1].content
            events=[e async for e in service.events(ChatRequest(user_id=user,conversation_id=cid,message='你好'))]
            assert events[-1][0]=='done'
            assert calls==1
        finally:
            if not task.done(): task.cancel()
            await service.summary_worker.aclose();service.active.discard(cid);remove_case(cid)


def test_tool_marker_bounds_status_without_truncating_order_identifier():
    from app.context.layers import tool_marker
    from app.context.tokens import estimate_text
    marker=tool_marker(ToolMessage(content=json.dumps({'order_id':'1001','status':'异常'*500}),tool_call_id='c'),'query_order')
    assert estimate_text(marker)<=120
    assert '1001' in marker and '字段过长' in marker
