from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def test_chinese_is_counted_without_utf8_byte_inflation():
    from app.context.tokens import estimate_text
    assert estimate_text('中文订单') == 4
    assert estimate_text('abcd') == 1

def test_tool_schema_and_call_are_counted():
    from app.context.tokens import estimate_messages
    base = [HumanMessage(content='你好')]
    tools = [{'name':'query_order','parameters':{'type':'object','properties':{'order_id':{'type':'string'}}}}]
    assert estimate_messages(base,tools) > estimate_messages(base)
    called = AIMessage(content='',tool_calls=[{'id':'a','name':'query_order','args':{'order_id':'1001'}}])
    result = ToolMessage(content='结果'*100,tool_call_id='a')
    assert estimate_messages([called,result]) > estimate_messages(base)
