import copy
import json

import pytest
from pymilvus import DataType, MilvusClient, RRFRanker


class Client:
    create_schema = MilvusClient.create_schema
    prepare_index_params = MilvusClient.prepare_index_params

    def __init__(self):
        self.created = None
        self.calls = []
        self.description = None
        self.indexes = None

    def has_collection(self, name):
        assert name == 'knowledge_ch04'
        return self.description is not None

    def create_collection(self, **kwargs):
        self.created = kwargs

    def load_collection(self, name):
        assert name == 'knowledge_ch04'

    def describe_collection(self, name):
        return self.description

    def list_indexes(self, name, field_name):
        return [field_name]

    def describe_index(self, name, index_name):
        return self.indexes[index_name]

    def upsert(self, **kwargs):
        self.calls.append(kwargs)

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [[{'id': 9, 'distance': .75, 'entity': {
            'question': '退货？', 'answer': '七天', 'category': '售后',
            'text': '原文', 'section_path': '售后/期限', 'content_type': 'policy',
            'is_key_clause': True,
        }}]]

    hybrid_search = search


def store_and_client():
    from app.knowledge.hybrid_store import HybridMilvusStore
    client = Client()
    return HybridMilvusStore(client=client), client


def row():
    return dict(id=9, dense=[0.] * 1024, text='原文', question='问题', answer='答案',
                category='售后', section_path='', content_type='policy', is_key_clause=True)


def test_owned_milvus_client_has_a_finite_connection_and_rpc_timeout(monkeypatch):
    from app.knowledge import hybrid_store
    class TimedClient:
        def __init__(self, *, uri, timeout):
            self.timeout = timeout
    monkeypatch.setattr(hybrid_store, "MilvusClient", TimedClient)
    store = hybrid_store.HybridMilvusStore(uri="http://local.invalid")
    assert store.client.timeout == 20.0


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
def test_online_search_rpc_has_an_explicit_deadline(strategy):
    store, client = store_and_client()
    if strategy == "dense":
        store.dense_search([0.0] * 1024)
    elif strategy == "bm25":
        store.bm25_search("邮费")
    else:
        store.hybrid_search([0.0] * 1024, "邮费")
    assert client.calls[0]["timeout"] == 20.0


def test_schema_has_exact_caps_analyzer_bm25_and_both_indexes():
    store, client = store_and_client()
    store.ensure_collection()
    schema = client.created['schema'].to_dict()
    fields = {f['name']: f for f in schema['fields']}
    assert set(fields) == {'id', 'dense', 'sparse', 'text', 'question', 'answer', 'category', 'section_path', 'content_type', 'is_key_clause'}
    assert fields['id']['is_primary'] and not schema['auto_id']
    assert not schema['enable_dynamic_field']
    assert fields['dense']['params']['dim'] == 1024
    assert fields['sparse']['type'] == DataType.SPARSE_FLOAT_VECTOR
    assert fields['is_key_clause']['type'] == DataType.BOOL
    for name, cap in [('text', 65535), ('question', 8192), ('answer', 49152), ('category', 1024), ('section_path', 2048), ('content_type', 128)]:
        assert fields[name]['params']['max_length'] == cap
    assert fields['text']['params']['enable_analyzer'] is True
    assert json.loads(fields['text']['params']['analyzer_params']) == {'type': 'chinese'}
    fn = schema['functions'][0]
    assert int(fn['type']) == 1
    assert fn['input_field_names'] == ['text'] and fn['output_field_names'] == ['sparse']
    indexes = [i.to_dict() for i in client.created['index_params']]
    assert {(i['field_name'], i['index_type'], i['metric_type']) for i in indexes} == {
        ('dense', 'AUTOINDEX', 'COSINE'), ('sparse', 'SPARSE_INVERTED_INDEX', 'BM25')}


@pytest.mark.parametrize('corruption', ['dimension', 'cap', 'analyzer', 'function', 'primary', 'dynamic', 'metric', 'extra'])
def test_existing_incompatible_collection_is_rejected(corruption):
    store, client = store_and_client()
    store.ensure_collection()
    description = copy.deepcopy(client.created['schema'].to_dict())
    fields = {f['name']: f for f in description['fields']}
    client.indexes = {i.field_name: i.to_dict() for i in client.created['index_params']}
    if corruption == 'dimension': fields['dense']['params']['dim'] = 3
    if corruption == 'cap': fields['answer']['params']['max_length'] = 65535
    if corruption == 'analyzer': fields['text']['params']['analyzer_params'] = {'type': 'standard'}
    if corruption == 'function': description['functions'] = []
    if corruption == 'primary': fields['id']['is_primary'] = False
    if corruption == 'dynamic': description['enable_dynamic_field'] = True
    if corruption == 'metric': client.indexes['sparse']['metric_type'] = 'IP'
    if corruption == 'extra': description['fields'].append({'name': 'extra'})
    client.description = description
    with pytest.raises(RuntimeError, match='incompatible'):
        store.ensure_collection()


@pytest.mark.parametrize('field,cap', [('text', 65535), ('question', 8192), ('answer', 49152), ('category', 1024), ('section_path', 2048), ('content_type', 128)])
def test_utf8_caps_are_checked_before_upsert_and_report_chunk_id(field, cap):
    store, client = store_and_client()
    data = row()
    data[field] = '中' * (cap // 3) + 'a' * (cap % 3)
    store.upsert([data])
    data[field] += 'a'
    with pytest.raises(ValueError, match='9'):
        store.upsert([data])
    assert len(client.calls) == 1


def test_hybrid_has_top50_each_leg_rrf_metadata_and_typed_filter():
    from app.knowledge.hybrid_store import build_filter
    store, client = store_and_client()
    value = '售后" or id > 0 or category == "x\\'
    filt = build_filter(category=value, content_type='policy', is_key_clause=True)
    result = store.hybrid_search([0.] * 1024, '退货', 50, 10, filt)
    call = client.calls[-1]
    assert call['collection_name'] == 'knowledge_ch04'
    assert call['consistency_level'] == 'Strong'
    assert call['limit'] == 10 and isinstance(call['ranker'], RRFRanker)
    assert {r.anns_field for r in call['reqs']} == {'dense', 'sparse'}
    expected = 'category == ' + json.dumps(value, ensure_ascii=False) + ' and content_type == "policy" and is_key_clause == true'
    assert all(r.limit == 50 and r.expr == expected for r in call['reqs'])
    assert result[0]['id'] == 9 and result[0]['score'] == .75
    assert result[0]['section_path'] == '售后/期限' and result[0]['is_key_clause'] is True
    for method, data, field in [(store.dense_search, [0.] * 1024, 'dense'), (store.bm25_search, '退货', 'sparse')]:
        method(data, 50, filt)
        assert client.calls[-1]['filter'] == expected
        assert client.calls[-1]['anns_field'] == field
        assert client.calls[-1]['consistency_level'] == 'Strong'
        with pytest.raises((TypeError, ValueError)):
            method(data, 50, 'id > 0')
    with pytest.raises((TypeError, ValueError)):
        build_filter(is_key_clause='true')


def test_count_and_paginated_ids_use_strong_and_close_iterator():
    store, client = store_and_client()
    calls = []
    client.query = lambda **kw: calls.append(kw) or [{'count(*)': 3}]
    class Iterator:
        batches = iter([[{'id': 1}, {'id': 5}], [{'id': 8}], []])
        def next(self): return next(self.batches)
        def close(self): calls.append('closed')
    client.query_iterator = lambda **kw: calls.append(kw) or Iterator()
    assert store.count() == 3
    assert store.ids() == {1, 5, 8}
    assert all(c['consistency_level'] == 'Strong' for c in calls[:-1])
    assert calls[-1] == 'closed'
