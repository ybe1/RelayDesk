from langchain_core.messages import HumanMessage,ToolMessage,AIMessage
from app.workflow.graph import make_turn_input

async def test_two_turns_accumulate_without_reset():
    from langgraph.graph import StateGraph,START,END
    from langgraph.checkpoint.memory import InMemorySaver
    from app.workflow.contracts import WorkflowState
    builder=StateGraph(WorkflowState)
    builder.add_node('pass',lambda s:{})
    builder.add_edge(START,'pass'); builder.add_edge('pass',END)
    graph=builder.compile(checkpointer=InMemorySaver())
    config={'configurable':{'thread_id':'test'}}
    await graph.ainvoke(make_turn_input('订单1001',1,'u','a',[HumanMessage(content='订单1001',id='a:user')]),config)
    await graph.ainvoke(make_turn_input('它能退吗',1,'u','b',[HumanMessage(content='它能退吗',id='b:user')]),config)
    snap=await graph.aget_state(config)
    assert [m.content for m in snap.values['messages']]==['订单1001','它能退吗']

def test_tool_save_skips_mysql():
    from app.workflow.service import WorkflowChatService
    class Store:
        def session_factory(self): raise AssertionError('tool must not open SQL session')
    service=WorkflowChatService(None,Store(),model_factory=lambda:None)
    assert service._save(1,'t',ToolMessage(content='raw',tool_call_id='c')) is None
