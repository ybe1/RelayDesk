def test_audit_model_matches_contract():
    from app.db.models import ToolAuditLog
    table=ToolAuditLog.__table__
    assert not table.foreign_keys
    assert table.c.id.type.unsigned and table.c.retry_count.type.unsigned
    assert table.c.status.type.enums==['成功','失败','超时','校验拦下','权限拒绝']
    assert {i.name for i in table.indexes}=={'idx_conversation_id','idx_tool_name','idx_status'}


def test_audit_failure_does_not_escape(caplog):
    from app.repositories.tool_audits import ToolAuditRepository
    def broken(): raise ConnectionError('audit database offline')
    assert ToolAuditRepository(broken).write({'tool_name':'query_order'}) is False
    assert 'audit write failed' in caplog.text
