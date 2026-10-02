import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_assistant_text_and_prices_are_not_order_authority():
    from app.workflow.order_context import trusted_order_ids
    sources = trusted_order_ids('价格是9999元', [AIMessage(content='订单9999可以退')])
    assert sources == {'current':set(),'history':set()}


def test_tool_results_and_explicit_user_orders_are_authority():
    from app.workflow.order_context import trusted_order_ids
    history = [HumanMessage(content='订单1001物流'),AIMessage(content='',tool_calls=[
        {'name':'query_order','args':{'order_id':'1002'},'id':'a'}]),ToolMessage(tool_call_id='a',content=json.dumps(
        {'result':json.dumps({'order_id':'1002','status':'运输中'})}))]
    assert trusted_order_ids('订单 1003 能退吗',history) == {'current':{'1003'},'history':{'1001','1002'}}


def test_unpaired_or_faq_tool_output_cannot_claim_order_authority():
    from app.workflow.order_context import trusted_order_ids
    history = [AIMessage(content='',tool_calls=[{'name':'query_faq','args':{'keyword':'退货'},'id':'a'}]),
        ToolMessage(tool_call_id='a',content=json.dumps({'order_id':'9999'})),
        ToolMessage(tool_call_id='unknown',content=json.dumps({'order_id':'8888'}))]
    assert trusted_order_ids('它能退吗',history)['history'] == set()


def test_invented_slot_clears_question_and_order():
    from app.workflow.order_context import validate_resolution
    from app.workflow.routing_contracts import Resolution
    value = Resolution(question='订单9999能退吗', reference_status='resolved',scope='order_specific',
        order_id='9999',order_source='history')
    result = validate_resolution(value,'这个能退吗',[])
    assert result.question == '这个能退吗' and result.order_id is None
    assert result.reference_status == 'ambiguous'


def test_ambiguous_history_cannot_select_arbitrary_order():
    from app.workflow.order_context import validate_resolution
    from app.workflow.routing_contracts import Resolution
    history = [HumanMessage(content='看看订单1001和订单1002')]
    value = Resolution(question='订单1001能退吗',reference_status='resolved',scope='order_specific',
        order_id='1001',order_source='history')
    assert validate_resolution(value,'它能退吗',history).order_id is None


def test_recent_turns_keep_tool_pairs():
    from app.workflow.order_context import recent_turns
    messages = []
    for i in range(8):
        messages.extend([HumanMessage(content=f'订单{i}'),AIMessage(content='',tool_calls=[
            {'name':'query_order','args':{'order_id':str(i)},'id':str(i)}]),ToolMessage(content='ok',tool_call_id=str(i))])
    assert [m.content for m in recent_turns(messages,max_turns=2) if m.type=='human'] == ['订单6','订单7']
    assert len(recent_turns(messages,max_turns=2)) == 6


def test_demo_cards_share_tool_snapshot_and_receipt_facts():
    from app.workflow.order_context import demo_order_options, demo_order
    from app.tools.business import build_tools
    tools = {t.name:t for t in build_tools(lambda:None,1)}
    assert [x['order_id'] for x in demo_order_options()] == ['1001','1002','1003']
    for order_id in map(str,range(1000,1030)):
        order = demo_order(order_id)
        assert order == json.loads(tools['query_order'].invoke({'order_id':order_id}))
        assert order['mock'] is True and order['product_name'] and order['category']
        assert (order['days_since_receipt'] is not None) == (order['status']=='已完成')
def test_rewritten_question_cannot_smuggle_order_when_slot_is_empty():
    from app.workflow.routing_contracts import Resolution
    from app.workflow.order_context import validate_resolution
    result=Resolution(question='订单9999能退吗',reference_status='resolved',scope='order_specific',order_source='none')
    checked=validate_resolution(result,'这个能退吗',[])
    assert checked.question=='这个能退吗' and checked.reference_status=='ambiguous'
