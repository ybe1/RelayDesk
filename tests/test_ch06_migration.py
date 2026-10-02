def test_refund_model_matches_ddl_constraints():
    from app.db.models import RefundRequest
    table = RefundRequest.__table__
    assert set(table.c.keys()) == {'id','application_no','offer_id','conversation_id','order_id','reason','status','is_mock','snapshot','created_at'}
    assert table.c.reason.type.enums == ['不喜欢/不需要','商品质量问题','商品与描述不符','错发/漏发','其他']
    assert table.c.status.type.enums == ['submitted']
    assert {fk.target_fullname for fk in table.foreign_keys} == {'conversations.id','chat_action_offers.offer_id'}


def test_migration_check_does_not_mutate_schema():
    from scripts.migrate_ch06 import planned_operations
    missing = {'offer_actions':['handoff','ticket'],'refund_exists':False}
    assert len(planned_operations(missing)) == 2
    assert missing == {'offer_actions':['handoff','ticket'],'refund_exists':False}
