import pytest
from uuid import uuid4
from fastapi import FastAPI
from httpx import AsyncClient,ASGITransport
from tests.ch06_service_fixture import service_case,remove_case


async def test_pending_api_and_selection_errors(tmp_path):
    from app.api.chat import router,get_chat_service
    from app.schemas.chat import ChatRequest
    user='api-select-'+uuid4().hex[:12]
    cid=None
    try:
        async with service_case(tmp_path) as service:
            events=[e async for e in service.events(ChatRequest(user_id=user,message='我要退款'))]
            cid=events[0][1]['conversation_id']
            selection=next(data for kind,data in events if kind=='order_selection')
            app=FastAPI(); app.include_router(router)
            app.dependency_overrides[get_chat_service]=lambda:service
            async with AsyncClient(transport=ASGITransport(app=app),base_url='http://test') as client:
                pending=await client.get(f'/api/chat/{cid}/pending',params={'user_id':user})
                assert pending.status_code==200 and pending.json()['pending']['selection_id']==selection['selection_id']
                assert (await client.get(f'/api/chat/{cid}/pending',params={'user_id':'other'})).status_code==403
                body={'user_id':user,'conversation_id':cid,'turn_id':selection['turn_id'],'selection_id':selection['selection_id'],'order_id':'1001'}
                assert (await client.post('/api/chat/selection',json={**body,'order_id':'9999'})).status_code==422
                assert (await client.post('/api/chat/selection',json={**body,'turn_id':'old'})).status_code==409
                response=await client.post('/api/chat/selection',json=body)
                assert response.status_code==200 and '[DONE]' in response.text and 'workflow_summary' in response.text
                assert (await client.post('/api/chat/selection',json=body)).status_code==409
    finally:
        if cid: remove_case(cid)
