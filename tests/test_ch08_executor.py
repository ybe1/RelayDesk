import asyncio
import json
import pytest
from langchain.tools import tool
from app.tooling.contracts import CallContext,CatalogSnapshot,ToolDefinition,ToolPolicy


class Policy:
    def __init__(self,access='read'): self.access=access; self.enabled=True
    def lookup(self,key): return ToolPolicy(access=self.access,timeout_seconds=1,max_retries=2) if self.enabled else None


class Audits:
    def __init__(self): self.rows=[]
    def write(self,row): self.rows.append(row); return True


def engine(tool,access='read',context=None):
    from app.services.tool_executor import ToolExecutor
    definition=ToolDefinition.from_tool(tool)
    snapshot=CatalogSnapshot('v',(definition,)); policy=Policy(access); audits=Audits()
    return ToolExecutor(snapshot,policy=policy,audits=audits,context=context or CallContext(),backoff_seconds=0),policy,audits,definition


@pytest.mark.parametrize('failure',[TimeoutError,ConnectionError])
async def test_transient_read_retries_are_audited_once(failure):
    calls=[]
    @tool
    async def read_item(count:int)->dict:
        """Read an item."""
        calls.append(count); raise failure()
    runner,policy,audits,_=engine(read_item)
    result=await runner.execute({'name':'read_item','args':{'count':1},'id':'r'})
    assert len(calls)==3 and len(audits.rows)==1
    assert audits.rows[0]['retry_count']==2
    assert audits.rows[0]['status']==('超时' if failure is TimeoutError else '失败')
    assert audits.rows[0]['duration_ms']>=0 and 'error' in json.loads(result.content)


async def test_invalid_denied_and_not_found_do_not_retry():
    calls=[]
    @tool
    async def read_item(count:int)->dict:
        """Read an item."""
        calls.append(count); return {'found':False}
    runner,policy,audits,_=engine(read_item)
    bad=await runner.execute({'name':'read_item','args':{'count':'1'},'id':'a'})
    assert not calls and json.loads(bad.content)['error']['code']=='invalid_arguments'
    assert audits.rows[-1]['status']=='校验拦下'
    await runner.execute({'name':'read_item','args':{'count':1},'id':'b'})
    assert len(calls)==1 and audits.rows[-1]['retry_count']==0
    assert audits.rows[-1]['status']=='失败'
    policy.enabled=False
    await runner.execute({'name':'read_item','args':{'count':1},'id':'c'})
    assert len(calls)==1 and audits.rows[-1]['status']=='权限拒绝'


async def test_write_requires_bound_permit_and_never_retries():
    from app.tooling.permissions import issue_write_permit
    calls=[]
    @tool
    async def create_ticket(description:str)->dict:
        """Create a ticket."""
        calls.append(description); raise TimeoutError()
    context=CallContext(conversation_id=3,user_id='u',ticket_request=True)
    runner,_,audits,definition=engine(create_ticket,'write',context)
    call={'name':'create_ticket','args':{'description':'未收到包裹'},'id':'w'}
    assert runner.preflight(call).kind=='needs_confirmation'
    context.confirmation={'confirmed':True}
    await runner.execute(call)
    assert not calls and audits.rows[-1]['status']=='权限拒绝'
    context.confirmation=issue_write_permit(context,call,definition.definition_hash,'offer')
    result=await runner.execute(call)
    assert len(calls)==1 and audits.rows[-1]['retry_count']==0
    assert audits.rows[-1]['status']=='超时' and '待核实' in result.content
    changed={**call,'args':{'description':'换了描述'}}
    await runner.execute(changed)
    assert len(calls)==1


async def test_audit_exception_cannot_replace_success():
    @tool
    async def read_item()->dict:
        """Read item."""
        return {'status':'delivered','order_id':'1001','mock':True}
    runner,_,audits,_=engine(read_item)
    def broken(row): raise ConnectionError()
    audits.write=broken
    result=await runner.execute({'name':'read_item','args':{},'id':'ok'})
    assert json.loads(result.content)['result']['order_id']=='1001'


async def test_parent_cancellation_still_audits():
    entered=asyncio.Event()
    @tool
    async def read_item()->dict:
        """Read item."""
        entered.set(); await asyncio.Event().wait()
    runner,policy,audits,_=engine(read_item)
    runner.timeout_seconds=5
    task=asyncio.create_task(runner.execute({'name':'read_item','args':{},'id':'cancel'}))
    await entered.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert len(audits.rows)==1 and audits.rows[0]['status']=='失败'


async def test_sync_write_can_complete_late_but_is_not_retried():
    import threading
    from app.tooling.permissions import issue_write_permit
    release=threading.Event(); finished=threading.Event(); calls=[]
    @tool
    def create_ticket(description:str)->dict:
        """Create a slow ticket."""
        calls.append(description)
        release.wait(2); finished.set()
        return {'ticket_no':'TDEMO'}
    context=CallContext(conversation_id=3,user_id='u',ticket_request=True)
    runner,_,audits,definition=engine(create_ticket,'write',context)
    runner.timeout_seconds=.05
    call={'name':'create_ticket','args':{'description':'包裹未收到'},'id':'slowwrite'}
    context.confirmation=issue_write_permit(context,call,definition.definition_hash,'slowoffer')
    try:
        result=await runner.execute(call)
        assert '待核实' in result.content and not finished.is_set()
        assert audits.rows[-1]['retry_count']==0 and audits.rows[-1]['status']=='超时'
    finally: release.set()
    assert await asyncio.to_thread(finished.wait,2)
    assert len(calls)==1


async def test_read_connection_recovers_after_one_retry():
    calls=[]
    @tool
    async def read_item()->dict:
        """Read item."""
        calls.append(1)
        if len(calls)==1: raise ConnectionError()
        return {'mock':True,'status':'delivered'}
    runner,_,audits,_=engine(read_item)
    result=await runner.execute({'name':'read_item','args':{},'id':'retry'})
    assert json.loads(result.content)['result']['status']=='已签收'
    assert len(audits.rows)==1 and audits.rows[0]['retry_count']==1
