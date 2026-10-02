import json
from pathlib import Path


async def test_needs_review_cannot_keep_model_suggested_refund_action():
    from scripts.eval_ch06 import evaluate_assessment
    from tests.ch06_fakes import JSONSequenceModel
    case={'id':'R-review','intent':'退款退货','facts':{'days_since_receipt':2},'policy':'商品与包装完好才可退','expected_status':'needs_review'}
    model=JSONSequenceModel([{'kind':'answer','reason':'需核实','actions':['refund'],
        'refund_assessment':{'status':'needs_review','reason':'完好情况未知','evidence_numbers':[1]}}])
    result=await evaluate_assessment(model,case)
    assert result['passed'] and 'refund' not in result['actions']


def test_inventory_requires_all_labels_and_four_user_turns_per_dialogue():
    from scripts.eval_ch06 import load_cases,inventory
    cases=load_cases(Path('data/ch06'))
    counts=inventory(cases)
    assert counts['intent']==40 and counts['dialogues']>=6 and counts['dialogue_turns']>=24
    assert counts['expansion']>=8 and counts['assessment']>=3


def test_cached_success_skips_and_failed_case_can_retry():
    from scripts.eval_ch06 import should_run
    results={'x':{'passed':True},'y':{'passed':False}}
    assert not should_run('x',results,False) and not should_run('y',results,False)
    assert not should_run('x',results,True) and should_run('y',results,True)


def test_unreviewed_expansion_never_passes_quality_gate():
    from scripts.eval_ch06 import check_expansion_quality
    case={'question':'订单1002能退吗','must_expand':True}
    result=check_expansion_quality(case,['订单1002能退吗','火星天气'],None)
    assert not result['passed'] and result['review_status']=='pending'


def test_signature_changes_with_code_or_threshold(tmp_path):
    from scripts.eval_ch06 import evaluation_signature
    source=tmp_path/'logic.py'; source.write_text('old')
    old=evaluation_signature('model',0.75,[source])
    assert old!=evaluation_signature('model',0.8,[source])
    source.write_text('new')
    assert old!=evaluation_signature('model',0.75,[source])


async def test_assessment_uses_production_agent_and_rejects_invalid_reference():
    from scripts.eval_ch06 import evaluate_assessment
    from tests.ch06_fakes import JSONSequenceModel
    case={'id':'R-test','intent':'退款退货','facts':{'days_since_receipt':2,'item_intact':True},
          'policy':'完好可退','expected_status':'eligible'}
    model=JSONSequenceModel([{'kind':'answer','reason':'允许','actions':['refund'],
        'refund_assessment':{'status':'eligible','reason':'允许','evidence_numbers':[999]}}])
    result=await evaluate_assessment(model,case)
    assert not result['passed'] and result['offer_allowed'] is False
    assert any('本轮订单快照' in str(request) for request in model.requests)


def test_expansion_review_is_bound_to_actual_query_output():
    from scripts.eval_ch06 import check_expansion_quality,query_signature
    case={'question':'原问题','must_expand':True}
    queries=['原问题','退货条件']
    review={'query_signature':query_signature(queries),'facets_ok':True,'facts_preserved':True,'reason':'人工核对条件侧重点且无新增事实'}
    approved=check_expansion_quality(case,queries,review)
    assert approved['passed'] and approved['review_status']=='approved'
    assert check_expansion_quality(case,queries,approved['review'])['passed']
    assert not check_expansion_quality(case,['原问题','火星天气'],review)['passed']


async def test_production_assessment_requires_eligible_refund_action():
    from scripts.eval_ch06 import evaluate_assessment
    from tests.ch06_fakes import JSONSequenceModel
    case={'id':'R-positive','intent':'退款退货','facts':{'item_intact':True},'policy':'商品完好允许退货','expected_status':'eligible'}
    def model(actions): return JSONSequenceModel([{'kind':'answer','reason':'允许','actions':actions,
        'refund_assessment':{'status':'eligible','reason':'允许','evidence_numbers':[1]}}])
    assert (await evaluate_assessment(model(['refund']),case))['passed']
    assert not (await evaluate_assessment(model([]),case))['passed']
