from uuid import uuid4
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.db.session import SessionLocal
from app.services.chat import DatabaseChatStore
from langchain_core.messages import HumanMessage,AIMessage
from tests.ch06_service_fixture import remove_case

@pytest.fixture
def case():
    from app.api.conversations import router
    app=FastAPI(); app.include_router(router)
    user='sidebar-'+uuid4().hex
    store=DatabaseChatStore(SessionLocal)
    first=store.start_or_load(user,None);second=store.start_or_load(user,None)
    raw='订单1001历史原话。'+'全文。'*100
    store.append(first,HumanMessage(content=raw));store.append(first,AIMessage(content='原始回答。'*100))
    store.append(second,HumanMessage(content='第二个问题'))
    try:
        with TestClient(app) as client: yield client,user,first,second,raw
    finally: remove_case(first);remove_case(second)

def test_wrong_owner_gets_no_original_messages(case):
    client,user,first,second,raw=case
    response=client.get(f'/api/conversations/{first}/messages',params={'user_id':'other'})
    assert response.status_code==403 and raw not in response.text

def test_paginated_newest_first_and_full_original_history(case):
    client,user,first,second,raw=case
    page=client.get('/api/conversations',params={'user_id':user,'limit':1}).json()
    assert page['items'][0]['id']==second and page['next_cursor']
    old=client.get('/api/conversations',params={'user_id':user,'limit':1,'cursor':page['next_cursor']}).json()
    assert old['items'][0]['id']==first and old['next_cursor'] is None
    history=client.get(f'/api/conversations/{first}/messages',params={'user_id':user}).json()
    assert history['messages'][0]['content']==raw
    assert len(history['messages'][1]['content'])>80

def test_get_is_read_only_and_invalid_cursor_rejected(case):
    client,user,first,second,raw=case
    with SessionLocal() as session:
        from sqlalchemy import select,func
        from app.db.models import Ticket,RefundRequest
        before=(session.scalar(select(func.count()).select_from(Ticket)),session.scalar(select(func.count()).select_from(RefundRequest)))
    assert client.get('/api/conversations',params={'user_id':user,'cursor':'bad'}).status_code==422
    assert client.get(f'/api/conversations/{first}/messages',params={'user_id':user}).status_code==200
    with SessionLocal() as session:
        after=(session.scalar(select(func.count()).select_from(Ticket)),session.scalar(select(func.count()).select_from(RefundRequest)))
    assert before==after
