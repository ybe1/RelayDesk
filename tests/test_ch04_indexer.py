import pytest
from sqlalchemy import select

from app.db.models import KnowledgeChunk
from tests.test_knowledge_indexer import session_factory, FakeEmbedder, FakeStore


class Store(FakeStore):
    def ensure_collection(self): pass
    def count(self): return len(self.rows_by_id)
    def ids(self): return set(self.rows_by_id)


def snapshot(factory):
    with factory() as session:
        return [(r.id, r.vectorize_status, r.vector_id) for r in session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id))]


def test_all_rows_rebuild_and_recover_with_same_ids_without_ch03_changes(session_factory):
    from app.knowledge.ch04_indexer import rebuild_ch04
    with session_factory.begin() as session:
        row = session.get(KnowledgeChunk, 2)
        row.vectorize_status, row.vector_id = 'done', '2'
        row.section_path, row.content_type, row.is_key_clause = '售后/期限', 'policy', 1
    before = snapshot(session_factory)
    store, embedder = Store(), FakeEmbedder()
    def crash(): raise RuntimeError('after-upsert')
    with pytest.raises(RuntimeError, match='after-upsert'):
        rebuild_ch04(session_factory, embedder, store, batch_size=1, after_upsert=crash)
    assert store.ids() == {1}
    first = rebuild_ch04(session_factory, embedder, store, batch_size=1)
    second = rebuild_ch04(session_factory, embedder, store, batch_size=2)
    assert first == second == {'indexed': 2, 'sql_count': 2, 'milvus_count': 2, 'ids_match': True, 'failed': 0}
    assert store.ids() == {1, 2} and snapshot(session_factory) == before
    assert embedder.texts[0] == ['分类：配送\n问题：运费多少？\n答案：满 99 元包邮。']
    assert store.rows_by_id[2]['is_key_clause'] is True
    assert store.rows_by_id[2]['section_path'] == '售后/期限'
    assert store.rows_by_id[1]['section_path'] == ''


def test_rebuild_rejects_oversize_source_before_embedding_with_sql_id(session_factory):
    from app.knowledge.ch04_indexer import rebuild_ch04
    with session_factory.begin() as session:
        session.get(KnowledgeChunk, 1).answer = '中' * 16385
    embedder = FakeEmbedder()
    with pytest.raises(ValueError, match='1.*answer'):
        rebuild_ch04(session_factory, embedder, Store())
    assert embedder.texts == []


def test_rebuild_reports_stale_remote_ids_instead_of_false_parity(session_factory):
    from app.knowledge.ch04_indexer import rebuild_ch04
    store = Store()
    store.rows_by_id[999] = {}
    with pytest.raises(RuntimeError, match='parity'):
        rebuild_ch04(session_factory, FakeEmbedder(), store)
