from sqlalchemy.dialects import mysql

from app.db.models import Base, Conversation, FAQ, Message, Ticket


def test_ch02_schema_matches_required_mysql_columns():
    tables = Base.metadata.tables
    required_ch02_tables = {"conversations", "messages", "faq", "tickets"}
    assert required_ch02_tables <= set(tables)
    assert str(Conversation.__table__.c.id.type.compile(dialect=mysql.dialect())).upper() == "BIGINT UNSIGNED"
    assert str(Message.__table__.c.conversation_id.type.compile(dialect=mysql.dialect())).upper() == "BIGINT UNSIGNED"
    assert str(FAQ.__table__.c.question.type.compile(dialect=mysql.dialect())).upper() == "VARCHAR(512)"
    assert str(Ticket.__table__.c.ticket_no.type.compile(dialect=mysql.dialect())).upper() == "VARCHAR(32)"
    assert {fk.target_fullname for fk in Message.__table__.foreign_keys} == {"conversations.id"}
    assert {fk.target_fullname for fk in Ticket.__table__.foreign_keys} == {"conversations.id"}
    assert set(Conversation.__table__.c.status.type.enums) == {"进行中", "已转人工", "已结束"}
    assert set(Message.__table__.c.role.type.enums) == {"user", "assistant", "tool"}
    assert set(Ticket.__table__.c.ticket_type.type.enums) == {"售后", "投诉", "咨询"}
