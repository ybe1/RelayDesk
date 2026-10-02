from sqlalchemy.dialects import mysql

def test_summary_table_matches_user_ddl():
    from app.db.models import ConversationSummary, Conversation
    table = ConversationSummary.__table__
    assert 'idx_conv_upto' in {i.name for i in table.indexes}
    assert 'uk_conv_seq' in {c.name for c in table.constraints}
    assert not table.foreign_keys
    assert table.c.seq.nullable is False
    for name in ['summary','summary_upto_msg_id','layer1_from_msg_id']:
        assert Conversation.__table__.c[name].nullable
    assert table.c.id.type.unsigned
    assert 'UNSIGNED' in str(table.c.from_msg_id.type.compile(dialect=mysql.dialect()))
