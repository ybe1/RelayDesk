import pytest
from tests.ch06_fakes import JSONSequenceModel
from app.workflow.budget import TurnBudget, Limits


@pytest.mark.parametrize('small_output,expected_calls',[
    ({'intent':'订单','confidence':0.4},1),
    ({'intent':'订单','confidence':0.9},0),
    ({'intent':'订单'},1),
])
async def test_small_model_escalates_at_most_once(small_output,expected_calls):
    from app.workflow.intent import classify_with_escalation
    small=JSONSequenceModel([small_output])
    large=JSONSequenceModel([{'intent':'退款退货','confidence':0.95}])
    run=await classify_with_escalation(large,small,'我想退货',TurnBudget(Limits(),clock=lambda:0))
    assert len(small.requests)==1 and len(large.requests)==expected_calls
    assert run.upgraded==bool(expected_calls)
    assert run.decision.intent==('退款退货' if expected_calls else '订单')


async def test_default_large_and_uncertain_other():
    from app.workflow.intent import classify_with_escalation
    model=JSONSequenceModel([{'intent':'商品咨询','confidence':0.2}])
    run=await classify_with_escalation(model,None,'莫名其妙',TurnBudget(Limits(),clock=lambda:0))
    assert run.decision.intent=='其他' and run.model_tier=='large' and not run.upgraded


async def test_network_failure_does_not_become_other():
    from app.workflow.intent import classify_with_escalation
    model=JSONSequenceModel([ConnectionError('offline')])
    with pytest.raises(ConnectionError):
        await classify_with_escalation(model,None,'问题',TurnBudget(Limits(),clock=lambda:0))


async def test_invalid_large_fails_and_greeting_costs_nothing():
    from app.workflow.intent import classify_with_escalation
    model=JSONSequenceModel([{'intent':'订单','confidence':True}])
    budget=TurnBudget(Limits(),clock=lambda:0)
    greeting=await classify_with_escalation(model,None,'你好！',budget)
    assert greeting.model_tier=='rule' and not model.requests
    with pytest.raises(ValueError):
        await classify_with_escalation(model,None,'订单呢',budget)
