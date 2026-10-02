from uuid import uuid4
import asyncio
import pytest
from tests.ch06_service_fixture import service_case,remove_case


async def test_pending_selection_survives_process_reopen(tmp_path):
    from app.schemas.chat import ChatRequest,SelectionRequest
    user='resume-'+uuid4().hex[:12]
    cid=None
    try:
        async with service_case(tmp_path) as service:
            events=[e async for e in service.events(ChatRequest(user_id=user,message='我要退款'))]
            cid=events[0][1]['conversation_id']
            selection=next(data for kind,data in events if kind=='order_selection')
            used=selection['budget_usage']['model_calls']
            assert used==2 and not any(kind=='delta' for kind,_ in events)
        async with service_case(tmp_path,resuming=True) as service:
            pending=await service.pending_selection(user,cid)
            assert pending['selection_id']==selection['selection_id']
            req=SelectionRequest(user_id=user,conversation_id=cid,turn_id=selection['turn_id'],selection_id=selection['selection_id'],order_id='1001')
            events=[e async for e in service.selection_events(req)]
            summary=next(data for kind,data in events if kind=='workflow_summary')
            assert summary['usage']['model_calls']==5 and summary['stop_reason']=='answered'
            assert await service.pending_selection(user,cid) is None
            with pytest.raises(RuntimeError): await service.prepare_selection(req)
    finally:
        if cid: remove_case(cid)


async def test_disconnect_after_interrupt_keeps_pending_and_releases_lock(tmp_path):
    from app.schemas.chat import ChatRequest,SelectionRequest
    user='disconnect-'+uuid4().hex[:12]
    cid=None
    try:
        async with service_case(tmp_path) as service:
            stream=service.events(ChatRequest(user_id=user,message='我要退款'))
            async for kind,data in stream:
                if kind=='conversation': cid=data['conversation_id']
                if kind=='order_selection': break
            await stream.aclose()
            assert cid not in service.active
            selection=await service.pending_selection(user,cid)
            req=SelectionRequest(user_id=user,conversation_id=cid,turn_id=selection['turn_id'],selection_id=selection['selection_id'],order_id='1001')
            results=await asyncio.gather(service.prepare_selection(req),service.prepare_selection(req),return_exceptions=True)
            assert sum(isinstance(value,RuntimeError) for value in results)==1
            prepared=next(value for value in results if not isinstance(value,BaseException))
            service.release(prepared)
    finally:
        if cid: remove_case(cid)


def test_waiting_time_is_excluded_but_consumed_budget_remains():
    from app.workflow.budget import TurnBudget,Limits,BudgetExceeded
    clock=[0]
    old=TurnBudget(Limits(),clock=lambda:clock[0])
    old.model_calls=5; old.tokens=500
    clock[0]=30
    snapshot=old.snapshot()
    clock[0]=1030
    resumed=TurnBudget(Limits(),clock=lambda:clock[0],usage=snapshot)
    assert resumed.remaining_seconds()==90 and resumed.tokens==500
    resumed.reserve_llm(1,1); resumed.settle_llm({'total_tokens':2})
    with pytest.raises(BudgetExceeded,match='model_calls'): resumed.reserve_llm(1,1)
    clock[0]+=91
    with pytest.raises(BudgetExceeded,match='deadline'): resumed.check_deadline()


async def test_cancel_before_interrupt_marks_turn_cancelled(tmp_path):
    from app.schemas.chat import ChatRequest
    from tests.ch06_fakes import JSONSequenceModel
    entered=asyncio.Event()
    class BlockingIntent(JSONSequenceModel):
        async def ainvoke(self,messages,**kwargs):
            if self.requests:
                entered.set()
                await asyncio.Event().wait()
            return await super().ainvoke(messages,**kwargs)
    model=BlockingIntent([{'question':'我要退款','reference_status':'clear','scope':'order_specific','order_id':None,'order_source':'none'}])
    user='pre-cancel-'+uuid4().hex[:12]
    cid=None
    try:
        async with service_case(tmp_path) as service:
            service.model_factory=lambda:model
            async def consume():
                nonlocal cid
                async for kind,data in service.events(ChatRequest(user_id=user,message='我要退款')):
                    if kind=='conversation': cid=data['conversation_id']
            task=asyncio.create_task(consume())
            await asyncio.wait_for(entered.wait(),5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError): await task
            assert await service.pending_selection(user,cid) is None and cid not in service.active
            snapshot=await service.graph.aget_state(service._config(cid))
            assert snapshot.values['stop_reason']=='cancelled'
    finally:
        if cid: remove_case(cid)


async def test_selection_validation_and_new_message_invalidate_old_card(tmp_path):
    from app.schemas.chat import ChatRequest,SelectionRequest
    user='select-'+uuid4().hex[:12]
    cid=None
    try:
        async with service_case(tmp_path) as service:
            events=[e async for e in service.events(ChatRequest(user_id=user,message='我要退款'))]
            cid=events[0][1]['conversation_id']
            selection=next(data for kind,data in events if kind=='order_selection')
            args=dict(user_id=user,conversation_id=cid,turn_id=selection['turn_id'],selection_id=selection['selection_id'],order_id='1001')
            with pytest.raises(PermissionError): await service.prepare_selection(SelectionRequest(**{**args,'user_id':'other'}))
            with pytest.raises(ValueError): await service.prepare_selection(SelectionRequest(**{**args,'order_id':'9999'}))
            with pytest.raises(RuntimeError): await service.prepare_selection(SelectionRequest(**{**args,'turn_id':'old'}))
            prepared=await service.prepare_selection(SelectionRequest(**args))
            with pytest.raises(RuntimeError): await service.prepare_selection(SelectionRequest(**args))
            service.release(prepared)
            await service.prepare(ChatRequest(user_id=user,conversation_id=cid,message='你好'))
            service.active.clear()
            assert await service.pending_selection(user,cid) is None
            with pytest.raises(RuntimeError): await service.prepare_selection(SelectionRequest(**args))
    finally:
        if cid: remove_case(cid)
