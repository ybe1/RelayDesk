import pytest


@pytest.mark.parametrize('confidence', [True, '0.9', -0.1, 1.1, float('nan'), float('inf')])
def test_confidence_rejects_invalid_values(confidence):
    from app.workflow.routing_contracts import IntentDecision
    with pytest.raises(ValueError): IntentDecision(intent='物流', confidence=confidence)


def test_intent_has_exact_two_fields_and_other():
    from app.workflow.routing_contracts import IntentDecision
    assert IntentDecision(intent='其他', confidence=0.5).model_dump() == {'intent':'其他','confidence':0.5}
    with pytest.raises(ValueError): IntentDecision(intent='物流', confidence=0.9, reason='extra')


@pytest.mark.parametrize('queries', [[], ['', 'x'], ['x']*4, ['x'*513]])
def test_queries_are_bounded(queries):
    from app.workflow.routing_contracts import ExpandedQueries
    with pytest.raises(ValueError): ExpandedQueries(queries=queries)


def test_refund_requires_evidence_for_eligibility():
    from app.workflow.routing_contracts import RefundAssessment
    with pytest.raises(ValueError): RefundAssessment(status='eligible',reason='yes',evidence_numbers=[])

