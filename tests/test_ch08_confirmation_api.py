import pytest
from pydantic import ValidationError


def test_confirmation_body_cannot_replace_preview_parameters():
    from app.schemas.chat import TicketDecisionRequest
    with pytest.raises(ValidationError):
        TicketDecisionRequest(user_id='u',conversation_id=1,turn_id='t',confirmation_id='c',decision='confirm',description='篡改')


def test_confirmation_body_is_strict_and_bounded():
    from app.schemas.chat import TicketDecisionRequest
    assert TicketDecisionRequest(user_id='u',conversation_id=1,turn_id='t',confirmation_id='c',decision='cancel').decision=='cancel'
    with pytest.raises(ValidationError):
        TicketDecisionRequest(user_id='u',conversation_id='1',turn_id='t',confirmation_id='c',decision='confirm')

@pytest.mark.asyncio
@pytest.mark.parametrize('exception,status',[(PermissionError,403),(LookupError,404),(RuntimeError,409)])
async def test_confirmation_endpoint_maps_ownership_and_stale_errors(exception,status):
    import httpx
    from fastapi import FastAPI
    from app.api.chat import router,get_chat_service
    class Service:
        async def prepare_ticket_decision(self,req): raise exception('test')
    app=FastAPI();app.include_router(router)
    app.dependency_overrides[get_chat_service]=lambda:Service()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        result=await client.post('/api/chat/ticket-confirmation',json={'user_id':'u','conversation_id':1,
            'turn_id':'t','confirmation_id':'c','decision':'confirm'})
    assert result.status_code==status
