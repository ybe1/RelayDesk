import json
from pathlib import Path


def test_inventory_requires_all_labels_and_four_user_turns_per_dialogue():
    from scripts.eval_ch06 import load_cases,inventory
    cases=load_cases(Path('data/ch06'))
    counts=inventory(cases)
    assert counts['intent']==40 and counts['dialogues']==6 and counts['dialogue_turns']>=24
    assert counts['expansion']>=8 and counts['assessment']>=3


def test_cached_success_skips_and_failed_case_can_retry():
    from scripts.eval_ch06 import should_run
    results={'x':{'passed':True},'y':{'passed':False}}
    assert not should_run('x',results,False) and not should_run('y',results,False)
    assert not should_run('x',results,True) and should_run('y',results,True)
