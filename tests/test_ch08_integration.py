import json
from uuid import uuid4
import pytest
from langchain_core.tools import tool
from app.tooling.registry import ToolRegistry
from app.tooling.policy import PolicyStore


@pytest.mark.asyncio
async def test_new_registration_is_available_next_chat_without_service_restart(tmp_path):
    from tests.ch06_service_fixture import service_case,remove_case
    from tests.test_ch05_agent_graph import GraphModel
    from app.schemas.chat import ChatRequest
    policy_path=tmp_path/'policy.json';policy_path.write_text('{}')
    registry=ToolRegistry(PolicyStore(policy_path))
    @tool
    def query_demo_note(order_id:str)->dict:
        """查询演示订单备注。"""
        return {'order_id':order_id,'note':'请轻放'}
    cid=None;user='ch08-hot-'+uuid4().hex[:10]
    try:
        async with service_case(tmp_path) as service:
            service.tool_registry=registry
            model=GraphModel([{'kind':'answer','reason':'等待信息'}])
            service.model_factory=lambda:model
            first=[e async for e in service.events(ChatRequest(user_id=user,message='帮我建个工单'))]
            cid=first[0][1]['conversation_id']
            assert 'query_demo_note' not in model.advertised
            policy_path.write_text(json.dumps({'builtin::query_demo_note':{'access':'read'}}))
            registry.register_builtin(lambda ctx:query_demo_note)
            model=GraphModel([{'kind':'tools','tool_calls':[{'name':'query_demo_note','args':{'order_id':'1001'},'id':'note'}],'reason':'查询备注'},
                {'kind':'answer','reason':'说明备注'}])
            service.model_factory=lambda:model
            second=[e async for e in service.events(ChatRequest(user_id=user,conversation_id=cid,message='订单1001包装破损，先查备注'))]
            assert 'query_demo_note' in model.advertised
            statuses=[data for kind,data in second if kind=='tool_status']
            assert any(d.get('tool_name')=='query_demo_note' and d.get('phase')=='success' for d in statuses)
    finally:
        if cid: remove_case(cid)

from tests.test_ch08_mcp import mcp_pair

@pytest.mark.asyncio
async def test_chat_logistics_runs_through_real_mcp_and_audits_server(tmp_path,mcp_pair):
    from langchain_core.messages import AIMessage
    from tests.ch06_service_fixture import service_case,remove_case
    from tests.test_ch05_agent_graph import GraphModel
    from app.schemas.chat import ChatRequest
    from app.tooling.builtins import bind_builtins
    from app.db.session import SessionLocal
    from app.db.models import ToolAuditLog
    from sqlalchemy import select
    registry=ToolRegistry(PolicyStore('config/tool_policy.json'));registry.register_builtin(bind_builtins)
    cid=None;user='ch08-mcp-chat-'+uuid4().hex[:8]
    try:
        async with service_case(tmp_path) as service:
            service.tool_registry=registry;service.mcp_discovery=mcp_pair.discovery
            model=GraphModel([
                AIMessage(content=json.dumps({'question':'订单1001物流到哪了','reference_status':'clear','scope':'general','order_id':'1001','order_source':'current'})),
                AIMessage(content=json.dumps({'intent':'物流','confidence':0.99})),
                {'kind':'tools','tool_calls':[{'name':'query_logistics','args':{'order_id':'1001'},'id':'logistics'}],'reason':'查物流轨迹'},
                {'kind':'answer','reason':'按工具结果回答'}])
            service.model_factory=lambda:model
            events=[e async for e in service.events(ChatRequest(user_id=user,message='订单1001物流到哪了'))]
            cid=events[0][1]['conversation_id']
            with SessionLocal() as session:
                audit=session.scalar(select(ToolAuditLog).where(ToolAuditLog.conversation_id==cid))
                assert audit.tool_source=='mcp' and audit.mcp_server=='logistics' and audit.status=='成功'
            assert 'query_warranty' in model.advertised
    finally:
        if cid: remove_case(cid)


def test_production_tooling_registers_builtin_and_both_mcp_servers():
    from app.tooling.runtime import create_runtime_tools
    from app.tooling.contracts import CallContext
    registry,discovery=create_runtime_tools()
    assert set(discovery.connections)=={'logistics','after_sales'}
    snapshot=registry.snapshot(CallContext(session_factory=lambda:None,conversation_id=1))
    assert 'query_logistics' not in {d.name for d in snapshot.definitions}
    assert 'create_ticket' in {d.name for d in snapshot.definitions}

@pytest.mark.asyncio
async def test_cached_faq_still_passes_engine_and_audit():
    from langchain_core.messages import HumanMessage
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import TurnContext,AgentDependencies
    from app.workflow.budget import TurnBudget,Limits
    from app.tooling.contracts import CallContext
    from app.services.tool_executor import ToolExecutor
    from tests.test_ch08_executor import Audits
    from tests.test_ch05_agent_graph import GraphModel
    @tool
    def query_faq(keyword:str)->dict:
        """检索FAQ。"""
        raise AssertionError('must reuse gated evidence')
    audits=Audits();executor=ToolExecutor([query_faq],context=CallContext(),audits=audits)
    model=GraphModel([{'kind':'tools','tool_calls':[{'name':'query_faq','args':{'keyword':'退货'},'id':'faq'}],'reason':'查询'},
        {'kind':'answer','reason':'证据充分'}])
    context=TurnContext(AgentDependencies(model,[query_faq],executor),TurnBudget(Limits(tokens=100000)))
    result=await build_agent_graph().ainvoke({'messages':[HumanMessage(content='退货政策')],
        'turn_id':'cached-ch08','gate_allowed':True,'node_trace':[],
        'evidence':[{'id':1,'n':1,'answer':'七日可申请'}],'citations':[]},context=context)
    assert result['stop_reason']=='answered'
    assert len(audits.rows)==1 and audits.rows[0]['status']=='成功'


@pytest.mark.asyncio
async def test_normalized_faq_result_keeps_citations():
    from langchain_core.messages import HumanMessage
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import TurnContext,AgentDependencies
    from app.workflow.budget import TurnBudget,Limits
    from app.services.tool_executor import ToolExecutor
    from tests.test_ch08_executor import Audits
    from tests.test_ch05_agent_graph import GraphModel
    @tool
    def query_faq(keyword:str)->dict:
        """检索FAQ。"""
        return {'sufficient':True,'evidence':[{'id':18,'question':'退货政策','answer':'七日可申请','section_path':'售后/退货'}]}
    model=GraphModel([{'kind':'tools','tool_calls':[{'name':'query_faq','args':{'keyword':'退货'},'id':'faq'}],'reason':'查询'},
        {'kind':'answer','reason':'证据充分'}])
    context=TurnContext(AgentDependencies(model,[query_faq],ToolExecutor([query_faq],audits=Audits())),TurnBudget(Limits(tokens=100000)))
    result=await build_agent_graph().ainvoke({'messages':[HumanMessage(content='退货政策')],
        'turn_id':'faq-ch08','node_trace':[],'evidence':[],'citations':[]},context=context)
    assert result['citations'][0]['chunk_id']==18

@pytest.mark.asyncio
async def test_bare_loop_rejected_ticket_is_audited():
    from langchain_core.messages import AIMessage,HumanMessage
    from app.workflow.bare_loop import run_bare_loop
    from app.workflow.contracts import AgentDependencies
    from app.workflow.budget import TurnBudget,Limits
    from app.tooling.builtins import bind_ticket
    from app.tooling.contracts import CallContext
    from app.services.tool_executor import ToolExecutor
    from tests.test_ch08_executor import Audits
    from tests.ch05_fakes import SequenceModel
    audits=Audits();ticket=bind_ticket(CallContext())
    call={'name':'create_ticket','args':{'description':'包裹破损','ticket_type':'售后'},'id':'bare-ticket'}
    model=SequenceModel([AIMessage(content='',tool_calls=[call]),AIMessage(content='需要确认')])
    await run_bare_loop([HumanMessage(content='投诉')],AgentDependencies(model,[ticket],ToolExecutor([ticket],audits=audits)),TurnBudget(Limits(tokens=100000)))
    assert len(audits.rows)==1 and audits.rows[0]['status']=='权限拒绝'


@pytest.mark.asyncio
async def test_agent_keeps_catalog_schema_fixed_in_all_phases():
    from tests.test_ch05_agent_graph import scenario
    _,model,_=await scenario([{'kind':'answer','reason':'回答'}])
    assert 'create_ticket' in model.advertised


@pytest.mark.asyncio
async def test_workflow_tool_audit_retains_conversation_id():
    from langchain_core.messages import HumanMessage
    from app.workflow.agent import build_agent_graph
    from app.workflow.contracts import TurnContext, AgentDependencies
    from app.workflow.budget import TurnBudget, Limits
    from app.tooling.contracts import CallContext
    from app.services.tool_executor import ToolExecutor
    from app.tools.business import build_tools
    from tests.test_ch08_executor import Audits
    from tests.test_ch05_agent_graph import GraphModel, tools_decision
    audits = Audits()
    tools = build_tools(lambda: None, 7)
    model = GraphModel([tools_decision("query_order"), {"kind": "answer", "reason": "已有订单信息"}])
    context = TurnContext(AgentDependencies(model, tools, ToolExecutor(
        tools, audits=audits, context=CallContext(conversation_id=7))), TurnBudget(Limits(tokens=100000)))
    result = await build_agent_graph().ainvoke({"messages": [HumanMessage(content="订单1001")],
        "conversation_id": 7, "turn_id": "audit-turn", "node_trace": [], "evidence": [], "citations": []},
        context=context)
    assert result["stop_reason"] == "answered"
    assert len(audits.rows) == 1
    assert audits.rows[0]["conversation_id"] == 7
    assert audits.rows[0]["status"] == "成功"
