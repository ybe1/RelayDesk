import asyncio
import logging
from dataclasses import dataclass
from uuid import uuid4
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from app.config import settings
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
    def __init__(self, graph, store, model_factory=None, tools_factory=build_tools, retriever=None,policy_retriever=None,
                 context_config=None,summary_repository=None,summary_worker=None,tool_registry=None,mcp_discovery=None,
                 confidence_config=None,evidence_check=None,observability=None):
        self.graph, self.store = graph, store
        if model_factory is None:
            from app.core.llm import get_chat_model
            model_factory = lambda: get_chat_model(streaming=True, disable_thinking=True, max_retries=0, stream_usage=True)
        self.model_factory, self.tools_factory, self.retriever = model_factory, tools_factory, retriever
        self.policy_retriever=policy_retriever
        self.confidence_config,self.evidence_check=confidence_config,evidence_check
        self.observability=observability
        self.tool_registry,self.mcp_discovery=tool_registry,mcp_discovery
        self.active = set()
        from app.context.budget import config_from_settings
        self.context_config=context_config or config_from_settings(settings)
        if summary_repository is None and callable(getattr(store,'session_factory',None)):
            from app.repositories.summaries import SummaryRepository
            summary_repository=SummaryRepository(store.session_factory)
        self.summary_repository=summary_repository
        if summary_worker is None and summary_repository is not None:
            from app.services.summarizer import SummaryWorker
            summary_worker=SummaryWorker(summary_repository,self.model_factory,self.context_config,LOGGER)
        self.summary_worker=summary_worker

    async def prepare(self, req):
        cid = await asyncio.to_thread(self.store.start_or_load, req.user_id, req.conversation_id)
        if cid in self.active:
            raise RuntimeError("conversation busy")
        self.active.add(cid)
        try:
            pending=await self._pending(cid)
            if pending:
                preview,values=pending
                messages=list(values.get('messages',[]))
                if preview.get('kind')=='ticket_confirmation':
                    from app.services.ticket_confirmation import TicketConfirmationService
                    from app.tooling.contracts import CallContext
                    from app.tooling.builtins import bind_ticket
                    from .ticket_flow import current_tool_call
                    service=TicketConfirmationService(self.store.session_factory)
                    await asyncio.to_thread(service.cancel,req.user_id,cid,preview['confirmation_id'])
                    bound=CallContext(conversation_id=cid,user_id=req.user_id,turn_id=preview['turn_id'],
                        session_factory=self.store.session_factory,ticket_request=True)
                    executor=ToolExecutor([bind_ticket(bound)],context=bound)
                    call=current_tool_call(values)
                    observation=await executor.reject(call,'客户继续新对话，旧工单预览已取消')
                    observation.id=f"{preview['turn_id']}:tool:{call['id']}"
                    messages.append(observation)
                await self.graph.aupdate_state(self._config(cid),{'stop_reason':'superseded','selection':{},'actions':[],
                    'messages':messages,'ticket_request':{'active':False,'user_messages':[]},'ticket_confirmation':{}},as_node='log')
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
                    if isinstance(selection,dict) and ('selection_id' in selection or selection.get('kind')=='ticket_confirmation'):
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
            if selection['turn_id']!=req.turn_id or selection.get('selection_id')!=req.selection_id: raise RuntimeError('stale selection')
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

    async def prepare_ticket_decision(self,req):
        cid=await asyncio.to_thread(self.store.start_or_load,req.user_id,req.conversation_id)
        if cid in self.active: raise RuntimeError('conversation busy')
        self.active.add(cid)
        try:
            pending=await self._pending(cid)
            if not pending: raise RuntimeError('no pending confirmation')
            preview,state=pending
            if (preview.get('kind')!='ticket_confirmation' or preview['turn_id']!=req.turn_id
                    or preview['confirmation_id']!=req.confirmation_id):
                raise RuntimeError('stale confirmation')
            return PreparedTurn(cid,req.turn_id,{'turn_id':req.turn_id,
                'confirmation_id':req.confirmation_id,'decision':req.decision},state)
        except BaseException:
            self.active.discard(cid)
            raise

    async def ticket_decision_events(self,req,prepared=None):
        from app.schemas.chat import ChatRequest
        prepared=prepared or await self.prepare_ticket_decision(req)
        original=ChatRequest(user_id=req.user_id,conversation_id=req.conversation_id,message=prepared.resume_state['raw_question'])
        events=self.events(original,prepared)
        try:
            async for event in events: yield event
        finally: await events.aclose()

    def release(self, prepared):
        self.active.discard(prepared.conversation_id)

    def _save(self, cid, turn_id, message):
        if message.type=='tool': return None
        from app.repositories.messages import append_once
        with self.store.session_factory() as session:
            row=append_once(session, cid, message.id or f"{turn_id}:{message.type}", message)
            session.commit()
            return row.id

    def _pool(self, cid, raw, source, reason, snapshot=None, event_key=None):
        from app.flywheel.pool import GapEvent,record_gap
        stored = ([{**hit,'retrieval_query':snapshot.get('query','')} for hit in list(snapshot.get('hits', []))[:3]]
                  if isinstance(snapshot, dict) else snapshot)
        return record_gap(GapEvent(cid,raw,source,reason,stored,event_key or f'{cid}:{source}:{raw}'),
                          self.store.session_factory)

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
        from app.context.budget import cumulative_token_limit,calculate_budget,ContextCapacityError
        from app.context.tokens import estimate_text
        cfg=self.context_config
        calls=max(settings.agent_max_model_calls,cfg.steps+5)
        budget = TurnBudget(Limits(model_calls=calls, tools=cfg.steps,
                                   tokens=cumulative_token_limit(cfg,calls), seconds=settings.agent_timeout_seconds),
                            usage=(prepared.resume_state or {}).get('budget_usage'))
        budget.context_config=cfg
        interruption = None
        pending_writes = set()
        saved_messages=[]
        message_db_ids=dict((prepared.resume_state or {}).get('message_db_ids',{}))
        async def write(function, *args):
            task = asyncio.create_task(asyncio.to_thread(function, *args))
            pending_writes.add(task)
            try:
                return await asyncio.shield(task)
            finally:
                if task.done():
                    pending_writes.discard(task)
        try:
            if estimate_text(req.message)>cfg.user_input:
                raise ContextCapacityError('用户输入超出上下文额度，请缩短后重试')
            if prepared.resume_value:
                history=list(prepared.resume_state.get('messages',[]))
                if 'confirmation_id' in prepared.resume_value:
                    confirmation=HumanMessage(content='确认提交工单' if prepared.resume_value['decision']=='confirm' else '取消提交工单',
                        id=f'{turn_id}:confirmation:{prepared.resume_value["confirmation_id"]}')
                else:
                    confirmation=HumanMessage(content='选择订单'+prepared.resume_value['order_id'],id=f'{turn_id}:selection:{prepared.resume_value["selection_id"]}')
                message_db_ids[confirmation.id]=await write(self._save,cid,turn_id,confirmation)
                saved_messages.append(confirmation)
            else:
                database = await asyncio.to_thread(self.store.load_history, cid)
                checkpoint=[]
                snapshot=None
                if getattr(self.graph,'checkpointer',None):
                    snapshot=await self.graph.aget_state(self._config(cid))
                    checkpoint=list(snapshot.values.get('messages',[]))
                    message_db_ids.update(snapshot.values.get('message_db_ids',{}))
                from .history import merge_history,close_missing_results
                history=merge_history(checkpoint,database)
                history.extend(close_missing_results(history))
                for old in database:
                    if old.additional_kwargs.get('db_id'):
                        message_db_ids[old.id]=old.additional_kwargs['db_id']
                message = HumanMessage(content=req.message, id=f"{turn_id}:user")
                message_db_ids[message.id]=await write(self._save, cid, turn_id, message)
                saved_messages.append(message)
            model = self.model_factory()
            from .ticket_flow import ticket_request_from_user
            previous=prepared.resume_state or (snapshot.values if snapshot else {})
            ticket_request=previous.get('ticket_request',{}) if prepared.resume_value else ticket_request_from_user(req.message,previous.get('ticket_request',{}))
            from app.tooling.contracts import CallContext
            from app.tooling.builtins import bind_ticket
            call_context=CallContext(conversation_id=cid,user_id=req.user_id,turn_id=turn_id,
                session_factory=self.store.session_factory,raw_question=req.message,budget=budget,model=model,
                ticket_request=bool(ticket_request.get('active')))
            call_context.ticket_sources=tuple([m.content for m in history if m.type=='human'
                and isinstance(m.content,str) and not m.additional_kwargs.get('context_materials')
                and not m.additional_kwargs.get('context_summary')]+ticket_request.get('user_messages',[]))
            catalog=None
            if self.tool_registry is not None:
                discovered=await self.mcp_discovery.discover() if self.mcp_discovery else ()
                catalog=self.tool_registry.snapshot(call_context,discovered)
                tools=catalog.model_tools()
            else:
                tools=self.tools_factory(self.store.session_factory,cid,raw_question=req.message,budget=budget,model=model)
                tools=[t for t in tools if t.name!='create_ticket']+[bind_ticket(call_context)]
            from app.context.budget import with_tool_prefix
            from app.context.compile import AGENT_SYSTEM
            from langchain_core.messages import SystemMessage
            from langchain_core.utils.function_calling import convert_to_openai_tool
            from .contracts import AgentDecision
            schemas=[convert_to_openai_tool(t) for t in [*tools,AgentDecision]]
            cfg=with_tool_prefix(cfg,[SystemMessage(content=AGENT_SYSTEM)],schemas)
            budget.context_config=cfg
            from app.context.contracts import LayerAnchors,HistoryProjection
            from app.context.layers import history_records,project_history,prepare_summary_batch
            from app.context.logging import record_history
            prior=[m for m in history if not (m.id or '').startswith(turn_id+':')]
            records=history_records(prior,message_db_ids)
            anchors,segments=await asyncio.to_thread(self.summary_repository.load,cid) if self.summary_repository else (LayerAnchors(),[])
            summary='\n'.join(s.content for s in segments)
            breakdown=calculate_budget(cfg,estimate_text(summary))
            if breakdown.history<breakdown.steady:
                raise ContextCapacityError('上下文预算不足，请开启新对话或调整模型窗口配置')
            projection=project_history(records,anchors,breakdown,summary)
            if self.summary_repository and projection.anchors.layer1_from!=anchors.layer1_from:
                moved=await asyncio.to_thread(self.summary_repository.advance_layer1,cid,anchors.layer1_from,projection.anchors.layer1_from)
                if not moved: raise RuntimeError('context anchors changed concurrently')
                LOGGER.info('layer1 demote conversation_id=%s %s→%s budget=%s',cid,anchors.layer1_from,projection.anchors.layer1_from,breakdown.layer1)
            record_history(cid,turn_id,projection,breakdown)
            if self.summary_worker and projection.layer2_tokens>breakdown.layer2:
                batch=prepare_summary_batch(cid,records,projection.anchors,segments)
                LOGGER.info('summary trigger conversation_id=%s from=%s upto=%s layer2=%s > budget=%s',cid,anchors.summary_upto+1,batch.upto_msg_id,projection.layer2_tokens,breakdown.layer2)
                self.summary_worker.submit(batch)
            async def save(msg):
                saved_messages.append(msg)
                db_id=await write(self._save, cid, turn_id, msg)
                if db_id is not None: message_db_ids[msg.id]=db_id
                return db_id
            async def pool(raw, source, reason, snapshot=None, event_key=None):
                await write(self._pool, cid, raw, source, reason, snapshot, event_key)
            async def offers(actions): return await write(self._offers, cid, turn_id, req.message, actions,context.refund_defaults)
            small_model=None
            if settings.intent_small_model:
                from app.core.llm import get_chat_model
                small_model=get_chat_model(model_name=settings.intent_small_model,disable_thinking=True,max_retries=0)
            context = TurnContext(AgentDependencies(model, tools, ToolExecutor(catalog or tools,context=call_context,policy=self.tool_registry.policy if self.tool_registry else None)), budget, save,
                                  retrieve=self.retriever, record_low_confidence=pool, issue_actions=offers,
                                  retrieve_queries=self.policy_retriever,intent_small_model=small_model,
                                  message_db_ids=message_db_ids)
            context.history_projection=projection
            context.context_config=cfg
            context.confidence_config=self.confidence_config
            context.evidence_check=self.evidence_check
            context.observability=self.observability
            state = Command(resume=prepared.resume_value) if prepared.resume_value else make_turn_input(req.message, cid, req.user_id, turn_id,
                                    [*history,message])
            if isinstance(state,dict): state['message_db_ids']=dict(message_db_ids)
            if isinstance(state,dict): state.update(ticket_request=ticket_request,ticket_confirmation={},tool_cursor=0,tool_observations=[])
            if isinstance(state,dict): state['context_anchors']={'summary_upto':projection.anchors.summary_upto,'layer1_from':projection.anchors.layer1_from}
            yield "conversation", {"conversation_id": cid}
            graph_config={"configurable": {"thread_id": f"conversation:{cid}"}, "recursion_limit": 50}
            if self.observability is not None:
                graph_config.update(self.observability.trace_config(cid,turn_id))
            from app.observability.spans import use_runtime
            with use_runtime(self.observability):
                async for namespace, event in self.graph.astream(state, graph_config,
                                                      context=context, stream_mode="custom", subgraphs=True):
                    yield event
            pending=await self._pending(cid)
            if pending:
                selection,values=pending
                yield 'ticket_confirmation' if selection.get('kind')=='ticket_confirmation' else 'order_selection',selection
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
                waiting=await self._pending(cid)
                if not waiting and getattr(self.graph,'checkpointer',None):
                    from .history import merge_history,close_missing_results
                    snapshot=await self.graph.aget_state(self._config(cid))
                    messages=merge_history(snapshot.values.get('messages',[]),saved_messages)
                    messages.extend(close_missing_results(messages))
                    await self.graph.aupdate_state(self._config(cid),{'messages':messages,
                        'message_db_ids':dict(message_db_ids)},as_node='log')
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
