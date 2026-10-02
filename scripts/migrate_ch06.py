"""Add demo refund persistence; existing rows are never rewritten or removed."""
import argparse
from sqlalchemy import inspect,text
from app.db.session import engine
from app.db.models import RefundRequest

ALTER_OFFERS = "ALTER TABLE chat_action_offers MODIFY COLUMN action ENUM('handoff','ticket','refund') NOT NULL"


def planned_operations(state):
    if state['offer_actions'] not in (['handoff','ticket'],['handoff','ticket','refund']):
        raise RuntimeError('unexpected offer ENUM; review required')
    operations = []
    if 'refund' not in state['offer_actions']: operations.append(ALTER_OFFERS)
    if not state['refund_exists']: operations.append('CREATE TABLE refund_requests')
    return operations


def _default(value):
    return str(value).strip("'").upper().replace('()','') if value is not None else None


def validate_refund_schema(conn):
    inspector = inspect(conn)
    columns = {c['name']:c for c in inspector.get_columns('refund_requests')}
    expected = RefundRequest.__table__
    if set(columns)!=set(expected.c.keys()): raise RuntimeError('refund column set mismatch')
    for field in expected.c:
        col = columns[field.name]
        actual_type = str(col['type'].compile(dialect=conn.dialect)).upper()
        expected_type = str(field.type.compile(dialect=conn.dialect)).upper()
        if actual_type!=expected_type or col['nullable']!=field.nullable:
            raise RuntimeError('refund type/nullability mismatch: '+field.name)
        want = _default(field.server_default.arg) if field.server_default else None
        if _default(col.get('default'))!=want:
            raise RuntimeError('refund default mismatch: '+field.name)
    indexes = {i['name']:i for i in inspector.get_indexes('refund_requests')}
    for index in expected.indexes:
        actual = indexes.get(index.name)
        if not actual or bool(actual['unique'])!=index.unique or actual['column_names']!=[c.name for c in index.columns]:
            raise RuntimeError('refund index mismatch: '+index.name)
    if inspector.get_pk_constraint('refund_requests')['constrained_columns']!=['id'] or not columns['id'].get('autoincrement'):
        raise RuntimeError('refund primary key mismatch')
    fks = {(tuple(f['constrained_columns']),f['referred_table'],tuple(f['referred_columns'])) for f in inspector.get_foreign_keys('refund_requests')}
    if fks!={(('offer_id',),'chat_action_offers',('offer_id',)),(('conversation_id',),'conversations',('id',))}:
        raise RuntimeError('refund foreign key mismatch')
    row = conn.execute(text("SELECT ENGINE,TABLE_COLLATION FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME='refund_requests'")).one()
    if row[0].upper()!='INNODB' or not row[1].startswith('utf8mb4_'):
        raise RuntimeError('refund engine/charset mismatch')


def migrate(apply=False):
    with engine.connect() as conn:
        inspector = inspect(conn)
        fields = {c['name']:c for c in inspector.get_columns('chat_action_offers')}
        state = {'offer_actions':fields['action']['type'].enums,'refund_exists':inspector.has_table('refund_requests')}
        if state['refund_exists']: validate_refund_schema(conn)
        operations = planned_operations(state)
        for sql in operations:
            print(('APPLY: ' if apply else 'PENDING: ')+sql)
            if apply:
                if sql.startswith('CREATE TABLE'): RefundRequest.__table__.create(conn)
                else: conn.execute(text(sql))
                conn.commit()
        if not operations: print('Ch06 refund schema is current')
        if apply: validate_refund_schema(conn)
        return operations


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    operations = migrate(args.apply)
    if operations and not args.apply: raise SystemExit(2)
