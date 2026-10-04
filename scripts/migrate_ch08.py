"""Explicit audit migration: inspect first; never replace a mismatching table."""
import argparse
from sqlalchemy import inspect,text
from sqlalchemy.dialects.mysql import dialect
from app.db.models import ToolAuditLog
from app.db.session import engine


def _default(value):
    return str(value).lower().replace('()','').strip("'") if value is not None else None


def expected_shape():
    table=ToolAuditLog.__table__
    return {'columns':{c.name:(str(c.type.compile(dialect=dialect())).upper(),c.nullable,
        _default(c.server_default.arg) if c.server_default else None,c.comment) for c in table.c},
        'indexes':{i.name:([c.name for c in i.columns],bool(i.unique)) for i in table.indexes},
        'primary_key':['id'],'autoincrement':True,'foreign_keys':[]}


def validate_shape(actual):
    if actual!=expected_shape(): raise RuntimeError('ch08 audit schema mismatch; existing table not changed')


def validate_schema(conn):
    inspector=inspect(conn); name=ToolAuditLog.__tablename__
    columns=inspector.get_columns(name)
    shape={'columns':{c['name']:(str(c['type'].compile(dialect=conn.dialect)).upper(),c['nullable'],_default(c.get('default')),c.get('comment')) for c in columns},
        'indexes':{i['name']:(i['column_names'],bool(i['unique'])) for i in inspector.get_indexes(name)},
        'primary_key':inspector.get_pk_constraint(name)['constrained_columns'],
        'autoincrement':bool(next(c for c in columns if c['name']=='id').get('autoincrement')),
        'foreign_keys':inspector.get_foreign_keys(name)}
    validate_shape(shape)
    row=conn.execute(text("SELECT ENGINE,TABLE_COLLATION,TABLE_COMMENT FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='tool_audit_logs'")).one()
    if row[0].upper()!='INNODB' or not row[1].startswith('utf8mb4_') or row[2]!='工具调用审计留痕':
        raise RuntimeError('ch08 engine/charset/comment mismatch')


def migrate(apply=False):
    with engine.connect() as conn:
        if not inspect(conn).has_table('tool_audit_logs'):
            if not apply:
                print('PENDING: tool_audit_logs'); return False
            conn.execute(text('SET NAMES utf8mb4'))
            ToolAuditLog.__table__.create(conn); conn.commit()
        validate_schema(conn)
        print('Ch08 audit schema is current'); return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true'); group.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    raise SystemExit(0 if migrate(args.apply) else 1)
