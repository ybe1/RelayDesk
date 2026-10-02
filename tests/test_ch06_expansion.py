import pytest
from tests.ch06_fakes import JSONSequenceModel
from app.workflow.budget import TurnBudget


async def test_original_first_deduplicated_and_no_faq_call():
    from app.workflow.expansion import expand_queries
    model=JSONSequenceModel([{'queries':['退货条件','退货条件','原问题']}])
    assert await expand_queries(model,'原问题',{},'退款退货',TurnBudget(clock=lambda:0))==['原问题','退货条件']
    assert await expand_queries(model,'FAQ',{},'商品咨询',TurnBudget(clock=lambda:0))==['FAQ']
    assert len(model.requests)==1


@pytest.mark.parametrize('value',[{'queries':['']},{'queries':['q'],'explanation':'extra'},{'queries':['a','b','c','d']}])
async def test_invalid_expansion_fails(value):
    from app.workflow.expansion import expand_queries
    with pytest.raises(ValueError): await expand_queries(JSONSequenceModel([value]),'问题',{},'售后',TurnBudget(clock=lambda:0))
