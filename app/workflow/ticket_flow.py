"""User-origin ticket authority and independent approval graph nodes."""
import re
import asyncio
from langgraph.types import interrupt
from app.services.ticket_confirmation import TicketConfirmationService
from app.tooling.permissions import issue_write_permit


def grounded_description(description,sources):
    if not isinstance(description,str) or not description.strip(): return False
    description=description.strip()
    if not any(description in source for source in sources): return False
    remainder=re.sub(r'(?:帮我|给我|替我|请|我要|我想|我需要)?\s*(?:创建|建立|建|提交|开|登记).{0,5}工单','',description)
    return len(re.sub(r'[\s，。！？,.!?呀啊吧谢谢你]','',remainder))>=4


def current_tool_call(state):
    return state['decision']['tool_calls'][state.get('tool_cursor',0)]


async def prepare_ticket(state, runtime):
    executor=runtime.context.agent.executor
    call=current_tool_call(state)
    definition=executor.snapshot.get(call['name'])
    service=TicketConfirmationService(executor.context.session_factory)
    preview=await asyncio.to_thread(service.prepare,executor.context,call,definition.definition_hash)
    return {'ticket_confirmation':preview,'node_trace':state.get('node_trace',[])+['prepare_ticket']}


def wait_ticket(state):
    reply=interrupt(state['ticket_confirmation'])
    preview=state['ticket_confirmation']
    if (not isinstance(reply,dict) or reply.get('confirmation_id')!=preview['confirmation_id']
            or reply.get('turn_id')!=preview['turn_id'] or reply.get('decision') not in {'confirm','cancel'}):
        raise PermissionError('confirmation response does not match preview')
    return {'ticket_confirmation':{**preview,'decision':reply['decision']}}


async def execute_ticket(state,runtime):
    from .agent import persist
    context=runtime.context
    executor=context.agent.executor
    call=current_tool_call(state)
    preview=state['ticket_confirmation']
    bound=executor.context
    service=TicketConfirmationService(bound.session_factory)
    if preview['decision']=='cancel':
        await asyncio.to_thread(service.cancel,bound.user_id,bound.conversation_id,preview['confirmation_id'])
        observation=await executor.reject(call,'客户取消了工单提交')
    else:
        await asyncio.to_thread(service.confirm,bound.user_id,bound.conversation_id,preview['confirmation_id'])
        definition=executor.snapshot.get(call['name'])
        if definition is None:
            observation=await executor.reject(call,'确认恢复时工具已不可用')
        else:
            bound.confirmation=issue_write_permit(bound,call,definition.definition_hash,preview['confirmation_id'])
            observation=await executor.execute(call,before_attempt=context.budget.reserve_tool)
    observation.id=f"{state['turn_id']}:tool:{call['id']}"
    await persist(context,observation)
    bound.ticket_request=False
    return {'messages':[observation],'tool_cursor':state.get('tool_cursor',0)+1,
        'ticket_confirmation':{},'ticket_request':{'active':False,'user_messages':[]},
        'tool_observations':[*state.get('tool_observations',[]),call['id']],
        'budget_usage':context.budget.snapshot(),'node_trace':state.get('node_trace',[])+['execute_ticket']}


def ticket_request_from_user(raw, previous):
    # Only direct user text reaches this boundary, never tool output or a
    # model-written question. Ambiguous mentions fail closed; preview still
    # requires a separate explicit confirmation before any write.
    plain=re.sub(r'[“「『"].*?[”」』"]', '', raw)
    if re.search(r'不要|不用|不必|取消|不建|不创建|不提交|算了|别.{0,8}工单',plain):
        return {'active':False,'user_messages':[]}
    quoted_or_question=bool(re.search(r'工具返回|系统提示|他说|她说|怎么|如何|什么意思',plain))
    explicit=not quoted_or_question and bool(re.search(
        r'(?:帮我|给我|替我|请|我要|我想|我需要).{0,10}(?:创建|建立|建|提交|开|登记).{0,5}工单',plain))
    active=explicit or bool(previous.get('active'))
    messages=[*previous.get('user_messages',[]),raw] if active else []
    return {'active':active,'user_messages':messages}
