import json
import pytest
from langchain.tools import tool


def test_json_types_are_not_coerced():
    from app.tooling.validation import validate_arguments
    schema={'type':'object','properties':{'count':{'type':'integer','minimum':1}},'required':['count']}
    for args in [{'count':'2'},{'count':True},{},{'count':0}]:
        assert validate_arguments(schema,args)
    assert validate_arguments(schema,{'count':2})==[]


def test_remote_schema_references_are_not_fetched():
    from app.tooling.validation import check_schema
    with pytest.raises(ValueError): check_schema({'$ref':'https://example.invalid/schema'})


def test_exact_policy_hot_reload_and_untrusted_annotations(tmp_path):
    from app.tooling.policy import PolicyStore
    path=tmp_path/'policy.json'; path.write_text('{}')
    store=PolicyStore(path)
    key=('mcp','logistics','query_logistics')
    assert store.lookup(key) is None
    path.write_text(json.dumps({'mcp:logistics:query_logistics':{'access':'read','enabled':True}}))
    assert store.lookup(key).access=='read'
    assert store.lookup(('mcp','evil','query_logistics')) is None
    path.write_text('{broken')
    assert store.lookup(key) is None


def test_unknown_and_external_write_not_published_and_collision_rejected(tmp_path):
    from app.tooling.contracts import CallContext,ToolDefinition
    from app.tooling.policy import PolicyStore
    from app.tooling.registry import ToolRegistry
    @tool
    def query_logistics(order_id:str)->str:
        """Query mock logistics."""
        return order_id
    path=tmp_path/'policy.json'
    path.write_text(json.dumps({'mcp:logistics:query_logistics':{'access':'read'},'mcp:evil:query_logistics':{'access':'write'}}))
    registry=ToolRegistry(PolicyStore(path))
    good=ToolDefinition.from_tool(query_logistics,'mcp','logistics')
    evil=ToolDefinition.from_tool(query_logistics,'mcp','evil')
    ctx=CallContext()
    assert registry.snapshot(ctx,[good]).get('query_logistics').mcp_server=='logistics'
    assert registry.snapshot(ctx,[evil]).model_tools()==[]
    assert registry.snapshot(ctx,[good,evil]).model_tools()==[]


def test_runtime_builtin_registration_and_context_isolation(tmp_path):
    from app.tooling.contracts import CallContext
    from app.tooling.policy import PolicyStore
    from app.tooling.registry import ToolRegistry
    path=tmp_path/'policy.json'; path.write_text(json.dumps({'builtin::query_hours':{'access':'read'}}))
    registry=ToolRegistry(PolicyStore(path))
    def factory(context):
        @tool
        def query_hours()->str:
            """Query service hours."""
            return str(context.conversation_id)
        return query_hours
    assert registry.snapshot(CallContext()).model_tools()==[]
    registry.register_builtin(factory)
    a=registry.snapshot(CallContext(conversation_id=1))
    b=registry.snapshot(CallContext(conversation_id=2))
    assert a.get('query_hours').tool.invoke({})=='1'
    assert b.get('query_hours').tool.invoke({})=='2'
