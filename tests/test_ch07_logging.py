import logging
from app.context.contracts import HistoryProjection,LayerAnchors,ContextConfig
from app.context.budget import calculate_budget
from langchain_core.messages import HumanMessage

def test_actual_messages_and_summary_logged(caplog):
    from app.context.logging import record_model,record_history
    p=HistoryProjection(LayerAnchors(2,4),(),(HumanMessage(content='原话',id='a'),),'订单1001未核实。',30,())
    b=calculate_budget(ContextConfig())
    with caplog.at_level(logging.INFO,logger='app.context'):
        record_history(1,'t',p,b)
        record_model(1,'t','decision',[HumanMessage(content='实际发出')],[],b)
    assert 'history_ctx' in caplog.text and '订单1001未核实' in caplog.text
    assert 'model_ctx' in caplog.text and '实际发出' in caplog.text
