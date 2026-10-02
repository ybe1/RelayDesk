from langchain_core.messages import AIMessage,ToolMessage

def test_checkpoint_success_not_marked_interrupted():
    from app.workflow.history import close_missing_results
    request=AIMessage(content='',id='a',tool_calls=[{'id':'c','name':'query_order','args':{'order_id':'1001'}}])
    result=ToolMessage(content='success',tool_call_id='c',id='r')
    assert close_missing_results([request,result])==[]

def test_recovery_closes_request_without_execution():
    from app.workflow.history import close_missing_results
    request=AIMessage(content='',id='a',tool_calls=[{'id':'c','name':'query_order','args':{}}])
    result=close_missing_results([request])
    assert len(result)==1 and result[0].tool_call_id=='c'
    assert 'interrupted' in result[0].content
