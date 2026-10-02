import json
from langchain_core.messages import AIMessage,AIMessageChunk


class JSONSequenceModel:
    def __init__(self,values):
        self.values,self.requests = iter(values),[]
        self.kwargs,self.schema = {},None
    def bind(self,**kwargs):
        self.kwargs.update(kwargs)
        return self
    def bind_tools(self,tools,**kwargs):
        self.schema = None
        self.finish_tool = any(getattr(t,'__name__','')=='AgentDecision' for t in tools)
        return self
    def with_structured_output(self,schema,**kwargs):
        self.schema = schema
        return self
    async def ainvoke(self,messages,**kwargs):
        self.requests.append((list(messages),dict(kwargs)))
        value = next(self.values)
        if isinstance(value,BaseException): raise value
        raw = AIMessage(content=json.dumps(value,ensure_ascii=False),usage_metadata={
            'input_tokens':100,'output_tokens':30,'total_tokens':130})
        if self.schema is not None:
            return {'raw':raw,'parsed':self.schema.model_validate(value),'parsing_error':None}
        if getattr(self,'finish_tool',False) and isinstance(value,dict) and 'kind' in value:
            raw.content = ''
            raw.tool_calls = [{'name':'AgentDecision','args':value,'id':'finish','type':'tool_call'}]
        return raw
    async def astream(self,messages,**kwargs):
        self.requests.append((list(messages),dict(kwargs)))
        yield AIMessageChunk(content='演示回答[1]',usage_metadata={'input_tokens':100,'output_tokens':5,'total_tokens':105})
def resolution_message(question,order_id=None):
    from langchain_core.messages import AIMessage
    import json
    return AIMessage(content=json.dumps({'question':question,'reference_status':'clear',
        'scope':'order_specific' if order_id else 'general','order_id':order_id,
        'order_source':'current' if order_id else 'none'},ensure_ascii=False))

