from dataclasses import replace
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from app.context.contracts import ContextConfig,HistoryRecord,LayerAnchors
from app.context.budget import calculate_budget

def test_layer2_preserves_user_and_raw_history():
    from app.context.layers import project_history
    raw='我报的是订单1001，不是1002。'
    records=[HistoryRecord(HumanMessage(content=raw,id='a:user'),1,'a'),
        HistoryRecord(AIMessage(content='答'*300,id='a:final'),2,'a')]
    p=project_history(records,LayerAnchors(0,2),calculate_budget(ContextConfig()),'')
    assert p.layer2[0].content==raw
    assert p.layer2[1].content=='答'*80+'（已截短）'
    assert records[1].message.content=='答'*300

def test_full_turn_demotion_with_tool_pair_and_size():
    from app.context.layers import project_history
    request=AIMessage(content='',id='a:request',tool_calls=[{'id':'c','name':'query_order','args':{'order_id':'1001'}}])
    records=[HistoryRecord(HumanMessage(content='查订单1001',id='a:user'),1,'a'),
        HistoryRecord(request,2,'a'),HistoryRecord(ToolMessage(content='结果'*2000,tool_call_id='c',id='a:tool'),2,'a'),
        HistoryRecord(AIMessage(content='完成',id='a:final'),3,'a'),
        HistoryRecord(HumanMessage(content='新问题',id='b:user'),4,'b'),
        HistoryRecord(AIMessage(content='新答案',id='b:final'),5,'b')]
    budget=replace(calculate_budget(ContextConfig()),layer1=100,layer2=1000)
    p=project_history(records,LayerAnchors(),budget,'')
    assert p.anchors.layer1_from==3
    assert [m.type for m in p.layer2]==['human','ai','tool','ai']
    assert p.layer2[2].tool_call_id=='c'
    assert len(p.layer2[2].content)<120
    assert [m.id for m in p.layer1]==['b:user','b:final']

def test_default_does_not_compress_22_turns():
    from app.context.layers import project_history
    records=[]
    for n in range(22):
        records += [HistoryRecord(HumanMessage(content='问题'*100,id=f'{n}:user'),n*2+1,str(n)),
                    HistoryRecord(AIMessage(content='回答'*300,id=f'{n}:final'),n*2+2,str(n))]
    p=project_history(records,LayerAnchors(),calculate_budget(ContextConfig()),'')
    assert p.anchors==LayerAnchors() and len(p.layer1)==44 and not p.layer2

def test_temporary_trimming_keeps_entire_turns_without_advancing_summary():
    from app.context.layers import project_history
    records=[]
    for n in range(4):
        records += [HistoryRecord(HumanMessage(content='问'*50,id=f'{n}:user'),n*2+1,str(n)),
                    HistoryRecord(AIMessage(content='答'*200,id=f'{n}:final'),n*2+2,str(n))]
    budget=replace(calculate_budget(ContextConfig()),layer1=300,layer2=150)
    p=project_history(records,LayerAnchors(),budget,'')
    assert p.anchors.summary_upto==0
    assert p.omitted_ranges
    assert len(p.layer2)%2==0
