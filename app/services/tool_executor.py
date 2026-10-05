"""The only tool dispatch boundary: validation, permissions, bounded I/O, audit."""
import asyncio
import json
import logging
from pathlib import Path
from time import monotonic
from langchain_core.messages import ToolMessage
from app.tooling.contracts import CallContext,CatalogSnapshot,ToolDefinition,ExecutionOutcome
from app.tooling.policy import PolicyStore
from app.tooling.permissions import WritePermit
from app.tooling.validation import validate_arguments
from app.tooling.results import error,normalize_result,serialize_result

LOGGER=logging.getLogger(__name__)
POLICY_PATH=Path(__file__).resolve().parents[2]/'config/tool_policy.json'


class ToolExecutor:
    def __init__(self,tools,timeout_seconds=None,backoff_seconds=.1,*,policy=None,audits=None,context=None):
        self.snapshot=tools if isinstance(tools,CatalogSnapshot) else CatalogSnapshot('bound',tuple(ToolDefinition.from_tool(t) for t in tools))
        self.registry={d.name:d.tool for d in self.snapshot.definitions}
        self.policy=policy or PolicyStore(POLICY_PATH)
        if audits is None:
            from app.repositories.tool_audits import ToolAuditRepository
            audits=ToolAuditRepository()
        self.audits=audits
        self.context=context or CallContext()
        self.timeout_seconds=timeout_seconds
        self.backoff_seconds=backoff_seconds

    def preflight(self,call):
        definition=self.snapshot.get(call.get('name',''))
        if definition is None: return ExecutionOutcome('denied',error('unknown_tool','未知或当前不可用的工具'))
        policy=self.policy.lookup(definition.key)
        if not policy: return ExecutionOutcome('denied',error('permission_denied','本地权限规则未允许此工具'))
        errors=validate_arguments(definition.parameters,call.get('args',{}))
        if errors: return ExecutionOutcome('invalid',error('invalid_arguments','工具参数不正确，请补齐或更正',fields=errors[:10]))
        if definition.name=='create_ticket' or policy.access=='write':
            if definition.key!=('builtin',None,'create_ticket') or policy.access!='write' or not self.context.ticket_request:
                return ExecutionOutcome('denied',error('permission_denied','客户尚未明确要求创建工单'))
            if self.context.ticket_sources is not None:
                from app.workflow.ticket_flow import grounded_description
                if not grounded_description(call['args'].get('description'),self.context.ticket_sources):
                    return ExecutionOutcome('invalid',error('invalid_arguments','缺少客户实际提供的问题描述。请追问具体问题，description引用客户原话，不得编造。'))
            permit=self.context.confirmation
            if not isinstance(permit,WritePermit) or not permit.allows(self.context,call,definition.definition_hash):
                return ExecutionOutcome('needs_confirmation',error('permission_denied','user_confirmation_required：需要用户确认工单预览'))
        return ExecutionOutcome('ready',{})

    async def _audit(self,call,payload,status,retries,started):
        definition=self.snapshot.get(call.get('name',''))
        content=serialize_result(payload)
        record={'conversation_id':self.context.conversation_id,'tool_call_id':str(call.get('id',''))[:64] or None,
            'tool_name':str(call.get('name',''))[:128], 'tool_source':definition.source if definition else 'builtin',
            'mcp_server':definition.mcp_server if definition else None,'arguments':call.get('args'),
            'result_summary':content[:4000], 'status':status,
            'error_message':str(payload.get('error',{}).get('message',''))[:512] or None,
            'retry_count':min(retries,255),'duration_ms':min(round((monotonic()-started)*1000),2**32-1)}
        try:
            await asyncio.shield(asyncio.wait_for(asyncio.to_thread(self.audits.write,record),timeout=2))
        except Exception:
            LOGGER.exception('tool audit failed tool=%s call_id=%s',record['tool_name'],record['tool_call_id'])

    async def reject(self,call,reason):
        started=monotonic(); payload=error('permission_denied',reason)
        await self._audit(call,payload,'权限拒绝',0,started)
        return self._message(call.get('id',''),payload)

    async def fail(self,call,code,reason):
        started=monotonic(); payload=error(code,reason)
        await self._audit(call,payload,'失败',0,started)
        return self._message(call.get('id',''),payload)

    async def execute(self,call,*,before_attempt=None):
        started=monotonic(); retries=0; status='失败'; payload=error('execution_failed','工具调用未完成')
        definition=self.snapshot.get(call.get('name',''))
        try:
            check=self.preflight(call)
            if check.kind!='ready':
                payload=check.payload
                status='校验拦下' if check.kind=='invalid' else '权限拒绝'
            else:
                policy=self.policy.lookup(definition.key)
                if policy is None:
                    payload=error('permission_denied','工具权限已撤销'); status='权限拒绝'
                else:
                    attempts=1+(policy.max_retries if policy.access=='read' and policy.retry_safe else 0)
                    for attempt in range(attempts):
                        if self.policy.lookup(definition.key)!=policy:
                            payload=error('permission_denied','工具权限已变更'); status='权限拒绝'; break
                        retries=attempt
                        try:
                            cache=self.context.trusted_results.get(definition.name)
                            if cache is not None and cache.get('args')==call.get('args',{}): raw=cache['result']
                            else:
                                if before_attempt: before_attempt()
                                timeout=self.timeout_seconds if self.timeout_seconds is not None else policy.timeout_seconds
                                raw=await asyncio.wait_for(definition.tool.ainvoke(call.get('args',{})),timeout)
                            payload=json.loads(serialize_result(normalize_result(definition,raw)))
                            status='失败' if 'error' in payload else '成功'
                            break
                        except (TimeoutError,ConnectionError):
                            if attempt+1==attempts: raise
                            await asyncio.sleep(self.backoff_seconds*(attempt+1))
        except PermissionError:
            status='权限拒绝'
            payload=error('permission_denied','工单确认已失效、已取消或与调用不匹配')
        except TimeoutError:
            status='超时'
            payload=error('timeout','写操作超时，结果待核实，请勿重复提交' if definition and definition.name=='create_ticket' else '查询超时，请稍后重试')
        except asyncio.CancelledError:
            payload=error('execution_failed','调用已中断；执行结果待核实')
            raise
        except Exception as exc:
            from app.workflow.budget import BudgetExceeded
            payload=error('execution_failed','工具执行失败，请稍后重试或人工核实')
            if isinstance(exc,BudgetExceeded): raise
            LOGGER.warning('tool execution failed name=%s type=%s',call.get('name'),type(exc).__name__)
        finally:
            await self._audit(call,payload,status,retries,started)
        return self._message(call.get('id',''),payload)

    @staticmethod
    def _message(call_id,payload):
        return ToolMessage(content=serialize_result(payload),tool_call_id=call_id)
