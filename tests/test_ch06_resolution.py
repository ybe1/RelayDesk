import pytest
from langchain_core.messages import HumanMessage
from tests.ch06_fakes import JSONSequenceModel
from app.workflow.budget import TurnBudget


async def test_model_invented_order_is_not_trusted():
    from app.workflow.resolution import resolve_question
    model=JSONSequenceModel([{'question':'订单9999能否退货','reference_status':'resolved',
        'scope':'order_specific','order_id':'9999','order_source':'history'}])
    result=await resolve_question(model,'这个能退吗',[],TurnBudget(clock=lambda:0))
    assert result.order_id is None and result.reference_status=='ambiguous'
    assert result.question=='这个能退吗'


async def test_valid_history_reference_and_single_request():
    from app.workflow.resolution import resolve_question
    model=JSONSequenceModel([{'question':'订单1001能否退货','reference_status':'resolved',
        'scope':'order_specific','order_id':'1001','order_source':'history'}])
    result=await resolve_question(model,'它能退吗',[HumanMessage(content='订单1001到哪了')],TurnBudget(clock=lambda:0))
    assert result.order_id=='1001' and len(model.requests)==1


async def test_invalid_format_is_ambiguous_but_network_failure_propagates():
    from app.workflow.resolution import resolve_question
    result=await resolve_question(JSONSequenceModel([{'question':'猜的'}]),'这个能退吗',[],TurnBudget(clock=lambda:0))
    assert result.reference_status=='ambiguous' and result.question=='这个能退吗'
    with pytest.raises(ConnectionError):
        await resolve_question(JSONSequenceModel([ConnectionError('offline')]),'问题',[],TurnBudget(clock=lambda:0))
