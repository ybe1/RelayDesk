"""Bounded in-process append-only summary jobs, independent of chat latency."""
import asyncio
import json
import re
from dataclasses import replace
from time import monotonic
from langchain_core.messages import SystemMessage,HumanMessage
from app.context.tokens import estimate_messages
from app.context.layers import group_turns,cap_tool_result
from app.core.summary_prompts import SUMMARY_PROMPT

def _source_text(batch): return '\n'.join(str(r.message.content) for r in batch.source)

def validate_summary(content,batch):
    if not isinstance(content,str) or len(content)>200: raise ValueError('summary length/type invalid')
    if content and len(content)<20: raise ValueError('summary too short')
    source=_source_text(batch)
    for identifier in re.findall(r'[A-Za-z]*\d{3,}[A-Za-z\d-]*',content):
        if identifier not in source: raise ValueError('summary identifier lacks batch source')
    if not content and (re.search(r'\d{3,}',source) or any(word in source for word in ['订单','商品','退货','退款','投诉','物流','核实'])):
        raise ValueError('empty summary omitted batch facts')
    return content

def summary_messages(batch):
    source=[{'role':r.message.type,'content':r.message.content} for r in batch.source]
    return [SystemMessage(content=SUMMARY_PROMPT),HumanMessage(content=json.dumps({
        '旧摘要（只作背景）':[s.content for s in batch.previous_segments],
        '本批原文':source},ensure_ascii=False))]

class SummaryWorker:
    def __init__(self,repository,model_factory,config,logger,max_concurrent=2):
        self.repository,self.model_factory,self.config,self.logger=repository,model_factory,config,logger
        self.tasks={}; self.semaphore=asyncio.Semaphore(max_concurrent); self.closing=False

    def submit(self,batch):
        if self.closing or batch.conversation_id in self.tasks or not batch.source: return False
        task=asyncio.create_task(self._run(batch))
        self.tasks[batch.conversation_id]=task
        task.add_done_callback(lambda done:self.tasks.pop(batch.conversation_id,None))
        return True

    def _parts(self,batch):
        # Only new full turns become sources. Prior immutable segments remain
        # background. Deterministic tool projection is permitted if a full turn
        # exceeds the summary request capacity; user text is never truncated.
        limit=self.config.window-self.config.output-max(450,(self.config.window+49)//50)
        current=[]; expected=batch.expected_summary_upto; parts=[]
        for turn in group_turns(batch.source):
            candidate=replace(batch,expected_summary_upto=expected,source=tuple(current)+turn,
                upto_msg_id=max(r.owner_msg_id for r in turn))
            if current and estimate_messages(summary_messages(candidate))>limit:
                part=replace(batch,expected_summary_upto=expected,source=tuple(current),upto_msg_id=max(r.owner_msg_id for r in current))
                parts.append(part); expected=part.upto_msg_id; current=[]
                candidate=replace(batch,expected_summary_upto=expected,source=turn,upto_msg_id=max(r.owner_msg_id for r in turn))
            if estimate_messages(summary_messages(candidate))>limit:
                turn=tuple(replace(r,message=cap_tool_result(r.message,self.config.tool_result)) for r in turn)
                candidate=replace(candidate,source=turn)
                if estimate_messages(summary_messages(candidate))>limit: raise ValueError('summary turn exceeds window')
            current.extend(turn)
        if current:
            parts.append(replace(batch,expected_summary_upto=expected,source=tuple(current),upto_msg_id=max(r.owner_msg_id for r in current)))
        return parts

    async def _run(self,batch):
        started=monotonic(); cid=batch.conversation_id
        try:
            async with self.semaphore:
                self.logger.info('summary start conversation_id=%s from=%s upto=%s',cid,batch.expected_summary_upto+1,batch.upto_msg_id)
                for part in self._parts(batch):
                    anchors,_=await asyncio.to_thread(self.repository.load,cid)
                    if anchors.summary_upto!=part.expected_summary_upto:
                        self.logger.info('summary skip conversation_id=%s reason=stale from=%s upto=%s seconds=%.3f',cid,part.expected_summary_upto+1,part.upto_msg_id,monotonic()-started)
                        return
                    async with asyncio.timeout(60):
                        response=await self.model_factory().ainvoke(summary_messages(part),
                            max_tokens=min(self.config.output,600),response_format={'type':'json_object'},
                            extra_body={'thinking':{'type':'disabled'}})
                    parsed=json.loads(response.content)
                    if not isinstance(parsed,dict) or set(parsed)!={'summary'}: raise ValueError('invalid summary JSON')
                    content=validate_summary(parsed['summary'],part)
                    row=await asyncio.to_thread(self.repository.append,cid,part.expected_summary_upto,part.upto_msg_id,content)
                    updated,_=await asyncio.to_thread(self.repository.load,cid)
                    committed=updated.summary_upto==part.upto_msg_id
                    self.logger.info('summary %s conversation_id=%s from=%s upto=%s seq=%s reason=%s seconds=%.3f usage=%s',
                        'done' if row else 'skip',cid,part.expected_summary_upto+1,part.upto_msg_id,
                        row.seq if row else None,'committed' if row else ('no_facts' if committed else 'stale'),
                        monotonic()-started,getattr(response,'usage_metadata',None))
        except asyncio.CancelledError:
            self.logger.warning('summary failed conversation_id=%s from=%s upto=%s reason=cancelled seconds=%.3f',cid,batch.expected_summary_upto+1,batch.upto_msg_id,monotonic()-started)
            raise
        except Exception as exc:
            self.logger.warning('summary failed conversation_id=%s from=%s upto=%s reason=%s seconds=%.3f',cid,batch.expected_summary_upto+1,batch.upto_msg_id,type(exc).__name__,monotonic()-started)

    async def aclose(self):
        self.closing=True
        tasks=list(self.tasks.values())
        if not tasks: return
        done,pending=await asyncio.wait(tasks,timeout=5)
        for task in pending: task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
