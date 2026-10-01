import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4
from langchain_core.messages import HumanMessage
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


class WorkflowChatService:
    def __init__(self, graph, store, model_factory=None, tools_factory=build_tools, retriever=None):
        self.graph, self.store = graph, store
        if model_factory is None:
            from app.core.llm import get_chat_model
            model_factory = lambda: get_chat_model(streaming=True, disable_thinking=True, max_retries=0, stream_usage=True)
        self.model_factory, self.tools_factory, self.retriever = model_factory, tools_factory, retriever
        self.active = set()

    async def prepare(self, req):
        cid = await asyncio.to_thread(self.store.start_or_load, req.user_id, req.conversation_id)
        if cid in self.active:
            raise RuntimeError("conversation busy")
        self.active.add(cid)
        return PreparedTurn(cid, uuid4().hex)

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

    def _offers(self, cid, turn_id, raw, actions):
        from app.repositories.chat_actions import issue_offer
        with self.store.session_factory() as session:
            rows = []
            for action in dict.fromkeys(actions):
                defaults = {"description": raw[:4000], "ticket_type": "投诉" if "投诉" in raw else "咨询"}
                offer = issue_offer(session, cid, turn_id, action, defaults)
                rows.append({"offer_id": offer.offer_id, "action": action, "defaults": defaults})
            session.commit()
            return rows

    async def events(self, req, prepared=None):
        prepared = prepared or await self.prepare(req)
        cid, turn_id = prepared.conversation_id, prepared.turn_id
        budget = TurnBudget(Limits(model_calls=settings.agent_max_model_calls, tools=settings.agent_max_tool_steps,
                                   tokens=settings.agent_max_tokens, seconds=settings.agent_timeout_seconds))
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
            history = await asyncio.to_thread(self.store.load_history, cid)
            message = HumanMessage(content=req.message, id=f"{turn_id}:user")
            await write(self._save, cid, turn_id, message)
            model = self.model_factory()
            tools = self.tools_factory(self.store.session_factory, cid, raw_question=req.message, budget=budget, model=model)
            async def save(msg): await write(self._save, cid, turn_id, msg)
            async def pool(raw, reason): await write(self._pool, cid, raw, reason)
            async def offers(actions): return await write(self._offers, cid, turn_id, req.message, actions)
            context = TurnContext(AgentDependencies(model, tools, ToolExecutor(tools)), budget, save,
                                  retrieve=self.retriever, record_low_confidence=pool, issue_actions=offers)
            state = make_turn_input(req.message, cid, req.user_id, turn_id,
                                    trim_turns([*history, message], settings.token_budget))
            yield "conversation", {"conversation_id": cid}
            async for namespace, event in self.graph.astream(state, {"configurable": {"thread_id": f"conversation:{cid}"}, "recursion_limit": 50},
                                                  context=context, stream_mode="custom", subgraphs=True):
                yield event
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
                if interruption and hasattr(self.graph, "aupdate_state"):
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
