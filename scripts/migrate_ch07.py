"""Explicit, repeatable two-step migration. MySQL DDL commits implicitly."""
import argparse
from sqlalchemy import inspect,text
from app.db.models import Conversation,ConversationSummary
from app.db.session import engine

COLUMNS=('summary','summary_upto_msg_id','layer1_from_msg_id')
AFTER=('status','summary','summary_upto_msg_id')

def planned_operations(state):
    operations=[]
    for name,after in zip(COLUMNS,AFTER):
        if name not in state['columns']:
            col=Conversation.__table__.c[name]
            kind='TEXT' if name=='summary' else 'BIGINT UNSIGNED'
            operations.append(f"ALTER TABLE conversations ADD COLUMN {name} {kind} NULL COMMENT '{col.comment}' AFTER {after}")
    if not state['summaries']: operations.append('CREATE TABLE conversation_summaries')
    return operations

def validate_columns(conn,table,fields,allow_missing=False):
    inspector=inspect(conn)
    columns={c['name']:c for c in inspector.get_columns(table.name)}
    for name in fields:
        if name not in columns:
            if allow_missing: continue
            raise RuntimeError('missing column: '+name)
        col,want=columns[name],table.c[name]
        actual=str(col['type'].compile(dialect=conn.dialect)).upper()
        expected=str(want.type.compile(dialect=conn.dialect)).upper()
        if actual!=expected or col['nullable']!=want.nullable:
            raise RuntimeError('ch07 type/nullability mismatch: '+name)
        if col.get('comment')!=want.comment:
            raise RuntimeError('ch07 comment mismatch: '+name)

def validate_summaries(conn):
    table=ConversationSummary.__table__
    inspector=inspect(conn)
    validate_columns(conn,table,table.c.keys())
    cols=inspector.get_columns(table.name)
    if {c['name'] for c in cols}!=set(table.c.keys()): raise RuntimeError('ch07 column set mismatch')
    indexes={i['name']:i for i in inspector.get_indexes(table.name)}
    for name,fields,unique in [('uk_conv_seq',['conversation_id','seq'],True),('idx_conv_upto',['conversation_id','upto_msg_id'],False)]:
        actual=indexes.get(name)
        if not actual or actual['column_names']!=fields or bool(actual['unique'])!=unique:
            raise RuntimeError('ch07 index mismatch: '+name)
    if inspector.get_foreign_keys(table.name): raise RuntimeError('unexpected summary foreign keys')
    if inspector.get_pk_constraint(table.name)['constrained_columns']!=['id']:
        raise RuntimeError('ch07 primary key mismatch')
    if not next(c for c in cols if c['name']=='id').get('autoincrement'):
        raise RuntimeError('ch07 id must autoincrement')
    created=next(c for c in cols if c['name']=='created_at')
    if str(created.get('default','')).lower().replace('()','')!='current_timestamp':
        raise RuntimeError('ch07 timestamp default mismatch')
    row=conn.execute(text("SELECT ENGINE,TABLE_COLLATION FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='conversation_summaries'")).one()
    if row[0].upper()!='INNODB' or not row[1].startswith('utf8mb4_'):
        raise RuntimeError('ch07 engine/charset mismatch')

def migrate(apply=False):
    with engine.connect() as conn:
        conn.execute(text('SET NAMES utf8mb4'))
        inspector=inspect(conn)
        validate_columns(conn,Conversation.__table__,COLUMNS,allow_missing=True)
        exists=inspector.has_table('conversation_summaries')
        if exists: validate_summaries(conn)
        operations=planned_operations({'columns':{c['name'] for c in inspector.get_columns('conversations')},'summaries':exists})
        for sql in operations:
            print(('APPLY: ' if apply else 'PENDING: ')+sql)
            if apply:
                if sql.startswith('CREATE TABLE'): ConversationSummary.__table__.create(conn)
                else: conn.execute(text(sql))
                conn.commit()
        if apply or not operations:
            validate_columns(conn,Conversation.__table__,COLUMNS)
            validate_summaries(conn)
            print('Ch07 schema is current')
        return operations

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check',action='store_true')
    group.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if migrate(args.apply) and not args.apply: raise SystemExit(2)
