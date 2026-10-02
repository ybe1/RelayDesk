from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from app.context.contracts import HistoryProjection,LayerAnchors

def test_materials_after_current_user_and_single_fixed_system():
    from app.context.compile import compile_agent_messages
    p=HistoryProjection(LayerAnchors(2,4),(),(),'用户询问订单1001。',0,())
    result=compile_agent_messages(p,'它能退吗',[{'n':1,'answer':'商品完好且包装齐全可退','question':'退货'}],{},[],'decision')
    assert sum(m.type=='system' for m in result)==1
    current=next(i for i,m in enumerate(result) if m.type=='human' and m.content=='它能退吗')
    assert '早期摘要' in result[current+1].content
    assert all(m.type!='system' for m in result[current+1:])

def test_prefix_is_same_in_decision_and_final():
    from app.context.compile import compile_agent_messages
    p=HistoryProjection(LayerAnchors(),(),(),'',0,())
    a=compile_agent_messages(p,'q',[],{},[],'decision')
    b=compile_agent_messages(p,'q',[],{},[],'final_answer')
    assert a[0].content==b[0].content

def test_current_tools_projected_without_mutating_checkpoint():
    from app.context.compile import compile_agent_messages
    from app.context.tokens import estimate_messages
    p=HistoryProjection(LayerAnchors(),(),(),'',0,())
    raw=ToolMessage(content='大'*5000,tool_call_id='c')
    result=compile_agent_messages(p,'q',[],{},[raw],'decision',tool_limit=1200)
    assert estimate_messages([result[-1]])<=1200
    assert raw.content=='大'*5000

def test_oversized_critical_evidence_is_not_silently_shortened():
    from app.context.compile import project_evidence
    rows,complete=project_evidence([{'n':1,'answer':'关键条件'*200,'is_key_clause':True}],180)
    assert not complete
    assert rows[0]['answer']=='证据条款超出输入额度，需要人工核实。'
