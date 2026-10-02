import pytest

def test_offline_inventory_never_claims_live_quality():
    from scripts.eval_ch07 import inventory,load_cases
    result=inventory(load_cases())
    assert result['turns_per_profile']>=22 and result['summaries']==10
    assert result['live_quality']=='not_run'

def test_missing_turn_or_summary_source_cannot_pass():
    from scripts.eval_ch07 import assess_dialogue,assess_summary
    assert not assess_dialogue({'turns':[]},'demo')['passed']
    result=assess_summary({'required':['1001'],'forbidden':['保证']},'保证订单9999到货')
    assert not result['passed']


def test_explicit_no_promise_is_not_a_prohibited_promise():
    from scripts.eval_ch07 import assess_dialogue
    report={'mode':'local','source_checks_passed':True,'turns':[{'index':i+1,'passed':True,'segments_immutable':True,
        'anchors':{'layer1_from':0},'answer':'订单1001尚未核实，手机号13800000000；不能保证送达时间。'} for i in range(22)]}
    assert assess_dialogue(report,'default')['passed']
