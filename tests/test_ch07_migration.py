import pytest

def test_matching_migration_is_idempotent():
    from scripts.migrate_ch07 import planned_operations
    assert planned_operations({'columns': {'summary','summary_upto_msg_id','layer1_from_msg_id'},'summaries':True}) == []

def test_both_steps_planned_in_order():
    from scripts.migrate_ch07 import planned_operations
    ops = planned_operations({'columns':set(),'summaries':False})
    assert len(ops)==4
    assert 'summary ' in ops[0]
    assert 'summary_upto_msg_id ' in ops[1]
    assert 'layer1_from_msg_id ' in ops[2]
    assert ops[3]=='CREATE TABLE conversation_summaries'
