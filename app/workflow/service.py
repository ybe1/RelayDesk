import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from app.config import settings
from app.services.chat import trim_turns
from app.tools.business import build_tools
from app.services.tool_executor import ToolExecutor
from .contracts import AgentDependencies, TurnContext
from .budget import TurnBudget, Limits
from .graph import make_turn_input

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedTurn:
    conversation_id: int
    turn_id: str
    resume_value: dict | None = None
    resume_state: dict | None = None


class WorkflowChatService:
    def __init__(self, graph, store, model_factory=None, tools_factory=build_tools, retriever=None,policy_retriever=None):
        self.graph, self.store = graph, store
        if model_factory is None:
            from app.core.llm import get_chat_model
            model_factory = lambda: get_chat_model(streaming=True, disable_thinking=True, max_retries=0, stream_usage=True)
        self.model_factory, self.tools_factory, self.retriever = model_factory, tools_factory, retriever
        self.policy_retriever=policy_retriever
        self.active = set()

    async def prepare(self, req):
        cid = await asyncio.to_thread(self.store.start_or_load, req.user_id, req.conversation_id)
        if cid in self.active:
            raise RuntimeError("conversation busy")
        self.active.add(cid)
        try:
            if await self._pending(cid):
                await self.graph.aupdate_state(self._config(cid),{'stop_reason':'superseded','selection':{},'actions':[]},as_node='log')
        except BaseException:
            self.active.discard(cid)
            raise
        return PreparedTurn(cid, uuid4().hex)

    @staticmethod
    def _config(cid): return {'configurable':{'thread_id':f'conversation:{cid}'},'recursion_limit':50}

    async def _pending(self,cid):
        if not getattr(self.graph,'checkpointer',None): return None
        snapshot=await self.graph.aget_state(self._config(cid),subgraphs=True)
        def locate(current,inherited):
            values={**inherited,**current.values}
            for task in current.tasks:
                if hasattr(task.state,'values'):
                    found=locate(task.state,values)
                    if found: return found
                for pause in task.interrupts:
                    selection=pause.value
                    if isinstance(selection,dict) and 'selection_id' in selection:
                        return ({**selection,'conversation_id':cid,'budget_usage':values.get('budget_usage',{}),'node_trace':values.get('node_trace',[])},values)
            return None
        return locate(snapshot,{})

    async def pending_selection(self,user_id,cid):
        await asyncio.to_thread(self.store.start_or_load,user_id,cid)
        pending=await self._pending(cid)
        return pending[0] if pending else None

    async def prepare_selection(self,req):
        cid=await asyncio.to_thread(self.store.start_or_load,req.user_id,req.conversation_id)
        if cid in self.active: raise RuntimeError('conversation busy')
        self.active.add(cid)
        try:
            pending=await self._pending(cid)
            if not pending: raise RuntimeError('no pending selection')
            selection,state=pending
            if selection['turn_id']!=req.turn_id or selection['selection_id']!=req.selection_id: raise RuntimeError('stale selection')
            if req.order_id not in {item['order_id'] for item in selection['options']}: raise ValueError('invalid candidate')
            return PreparedTurn(cid,req.turn_id,{'selection_id':req.selection_id,'order_id':req.order_id},state)
        except BaseException:
            self.active.discard(cid)
            raise

    async def selection_events(self,req,prepared=None):
        from app.schemas.chat import ChatRequest
        prepared=prepared or await self.prepare_selection(req)
        original=ChatRequest(user_id=req.user_id,conversation_id=req.conversation_id,message=prepared.resume_state['raw_question'])
        events=self.events(original,prepared)
        try:
            async for event in events: yield event
        finally: await events.aclose()

    def release(self, prepared):
        self.active.discard(prepared.conversation_id)

    def _close_pending(self, cid):
        from app.repositories.messages import close_pending_tools
        with self.store.session_factory() as session:
            close_pending_tools(session, cid)
            session.commit()

    def _save(self, cid, turn_id, message):
        from app.repositories.messages import append_once
        with self.store.session_factory() as session:
            append_once(session, cid, message.id or f"{turn_id}:{message.type}", message)
            session.commit()

    def _pool(self, cid, raw, reason):
        from app.repositories.low_confidence import insert_low_confidence
        with self.store.session_factory() as session:
            insert_low_confidence(session, cid, raw, "retrieval_low_conf", reason)
            session.commit()

    def _offers(self, cid, turn_id, raw, actions,refund_defaults=None):
        from app.repositories.chat_actions import issue_offer
        with self.store.session_factory() as session:
            rows = []
            for action in dict.fromkeys(actions):
                defaults = {"description": raw[:4000], "ticket_type": "投诉" if "投诉" in raw else "咨询"}
                if action=='refund':
                    if refund_defaults is None: continue
                    defaults=refund_defaults
                offer = issue_offer(session, cid, turn_id, action, defaults)
                rows.append({"offer_id": offer.offer_id, "action": action, "defaults": defaults})
            session.commit()
            return rows

    async def events(self, req, prepared=None):
        prepared = prepared or await self.prepare(req)
        cid, turn_id = prepared.conversation_id, prepared.turn_id
        budget = TurnBudget(Limits(model_calls=settings.agent_max_model_calls, tools=settings.agent_max_tool_steps,
                                   tokens=settings.agent_max_tokens, seconds=settings.agent_timeout_seconds),
                            usage=(prepared.resume_state or {}).get('budget_usage'))
        interruption = None
        pending_writes = set()
        async def write(function, *args):
            task = asyncio.create_task(asyncio.to_thread(function, *args))
            pending_writes.add(task)
            try:
                return await asyncio.shield(task)
            finally:
                if task.done():
                    pending_writes.discard(task)
        try:
            await asyncio.to_thread(self._close_pending, cid)
            if prepared.resume_value:
                confirmation=HumanMessage(content='选择订单'+prepared.resume_value['order_id'],id=f'{turn_id}:selection:{prepared.resume_value["selection_id"]}')
                await write(self._save,cid,turn_id,confirmation)
            else:
                history = await asyncio.to_thread(self.store.load_history, cid)
                message = HumanMessage(content=req.message, id=f"{turn_id}:user")
                await write(self._save, cid, turn_id, message)
            model = self.model_factory()
            tools = self.tools_factory(self.store.session_factory, cid, raw_question=req.message, budget=budget, model=model)
            async def save(msg): await write(self._save, cid, turn_id, msg)
            async def pool(raw, reason): await write(self._pool, cid, raw, reason)
            async def offers(actions): return await write(self._offers, cid, turn_id, req.message, actions,context.refund_defaults)
            small_model=None
            if settings.intent_small_model:
                from app.core.llm import get_chat_model
                small_model=get_chat_model(model_name=settings.intent_small_model,disable_thinking=True,max_retries=0)
            context = TurnContext(AgentDependencies(model, tools, ToolExecutor(tools)), budget, save,
                                  retrieve=self.retriever, record_low_confidence=pool, issue_actions=offers,
                                  retrieve_queries=self.policy_retriever,intent_small_model=small_model)
            state = Command(resume=prepared.resume_value) if prepared.resume_value else make_turn_input(req.message, cid, req.user_id, turn_id,
                                    trim_turns([*history, message], settings.token_budget))
            yield "conversation", {"conversation_id": cid}
            async for namespace, event in self.graph.astream(state, {"configurable": {"thread_id": f"conversation:{cid}"}, "recursion_limit": 50},
                                                  context=context, stream_mode="custom", subgraphs=True):
                yield event
            pending=await self._pending(cid)
            if pending:
                selection,values=pending
                yield 'order_selection',selection
                yield 'workflow_summary',{'node_trace':values.get('node_trace',[]),'stop_reason':'waiting','usage':selection['budget_usage']}
            yield "done", None
        except BaseException as exc:
            interruption = "cancelled" if isinstance(exc, (asyncio.CancelledError, GeneratorExit)) else type(exc).__name__
            LOGGER.warning("workflow interrupted conversation_id=%s turn_id=%s reason=%s usage=%s",
                           cid, turn_id, interruption, budget.snapshot())
            raise
        finally:
            # Persisted tool requests are closed even if stream/model cancellation
            # happens between its request and observation checkpoints.
            async def cleanup():
                if pending_writes:
                    await asyncio.shield(asyncio.gather(*pending_writes, return_exceptions=True))
                await asyncio.to_thread(self._close_pending, cid)
                waiting=await self._pending(cid)
                if interruption and not waiting and hasattr(self.graph, "aupdate_state"):
                    await self.graph.aupdate_state({"configurable": {"thread_id": f"conversation:{cid}"}},
                        {"stop_reason": interruption, "budget_usage": budget.snapshot(), "actions": []}, as_node="log")
            cleanup_task = asyncio.create_task(cleanup())
            cancelled_during_cleanup = False
            try:
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        cancelled_during_cleanup = True
                cleanup_task.result()
            except Exception:
                LOGGER.exception("Unable to close interrupted tools conversation_id=%s", cid)
            finally:
                self.release(prepared)
            if cancelled_during_cleanup:
                raise asyncio.CancelledError
