"""Ch07 labelled evaluation; offline inventory never claims live model quality."""
import argparse
import asyncio
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from uuid import uuid4

ROOT=Path('data/ch07')

def load_cases():
    return {'dialogues':json.loads((ROOT/'dialogues.json').read_text(encoding='utf-8-sig')),
        'contexts':[json.loads(line) for line in (ROOT/'context_cases.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()],
        'summaries':[json.loads(line) for line in (ROOT/'summary_cases.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]}

def inventory(cases):
    turns=cases['dialogues']['turns'];summaries=cases['summaries']
    if len(turns)<22 or len(summaries)!=10: raise ValueError('incomplete labelled cases')
    if len({c['id'] for c in summaries})!=10: raise ValueError('duplicate summary case')
    if any(not t.get('question') for t in turns): raise ValueError('empty dialogue question')
    if {c['id'] for c in cases['contexts']}!={'C01','C02','C03','C04'}: raise ValueError('context case inventory incomplete')
    return {'turns_per_profile':len(turns),'summaries':len(summaries),'live_quality':'not_run'}

def assess_summary(case,summary):
    missing=[term for term in case.get('required',[]) if term not in summary]
    forbidden=[term for term in case.get('forbidden',[]) if term in summary]
    valid_length=(summary=='' if case.get('expect_empty') else 20<=len(summary)<=200)
    return {'passed':not missing and not forbidden and valid_length,'missing':missing,'forbidden':forbidden,
        'length':len(summary),'semantic_review':'required'}

def context_turn_completed(status,answer):
    if status.get('stop_reason') in {'answered','clarify','smalltalk'}: return True
    return (status.get('stop_reason')=='fallback' and
        'confidence_gate' in status.get('node_trace',[]) and
        answer=='知识库证据不足，暂时无法确认，请补充信息或人工核实。')

def assess_dialogue(report,profile):
    turns=report.get('turns',[])
    complete=len(turns)>=22 and all(t.get('passed') for t in turns)
    final=turns[-1].get('answer','') if turns else ''
    recall='1001' in final and '核实' in final and '13800000000' in final
    summary=report.get('segments',[])
    prohibited=('保证明天','保证到账','保证送达','承诺明天','一定明天','保证退货成功')
    def positive_text(text):
        return re.sub(r'(?:不能|无法|不会|不)(?:保证|承诺)(?:明天|到账|送达|退货成功)','',text)
    violations=[t.get('index') for t in turns if any(p in positive_text(t.get('answer','')) for p in prohibited)]
    structural=bool(report.get('source_checks_passed')) and all(t.get('segments_immutable',False) for t in turns)
    if profile=='default':
        layers=not summary and all(t.get('anchors',{}).get('layer1_from',0)==0 for t in turns)
    else:
        initial=report.get('initial_budget',{})
        layers=bool(summary) and any(t.get('anchors',{}).get('layer1_from',0)>0 for t in turns) and (
            initial.get('history'),initial.get('layer1'),initial.get('layer2'))==(5650,3954,1695)
    return {'passed':bool(complete and recall and layers and structural and not violations),'complete':complete,'earliest_fact_recalled':recall,
        'layer_behavior':layers,'source_checks_passed':structural,'prohibited_promises':violations,
        'semantic_review':'required','live_quality':'not_verified'}

def save_report(path,report):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temporary.replace(path)

def signature(profile,mode,model):
    from importlib.metadata import version
    digest=hashlib.sha256(json.dumps({'profile':profile,'mode':mode,'model':model,
        'versions':{n:version(n) for n in ['langchain-core','langchain-openai','langgraph']}},sort_keys=True).encode())
    files=[*Path('app').rglob('*.py'),Path('app/static/index.html'),ROOT/'dialogues.json',ROOT/'summary_cases.jsonl',ROOT/'context_cases.jsonl',Path(__file__)]
    for path in sorted(files): digest.update(str(path).encode());digest.update(path.read_bytes())
    return digest.hexdigest()

class LocalModel:
    """Deterministic test double. Never labelled as real LLM quality."""
    def bind_tools(self,tools,**kwargs): return self
    async def ainvoke(self,messages,**kwargs):
        from langchain_core.messages import AIMessage
        system=messages[0].content
        if '事实摘要' in system:
            data=json.loads(messages[-1].content)
            source=json.dumps(data['本批原文'],ensure_ascii=False)
            summary='用户关注演示订单1001的物流，包裹未到并请求继续核实，当前诉求尚未确认已解决。'
            if '13800000000' in source: summary+='演示联系手机号13800000000。'
            return AIMessage(content=json.dumps({'summary':summary},ensure_ascii=False))
        if 'current_question' in messages[-1].content:
            data=json.loads(messages[-1].content); raw=data['current_question']
            if 'intent' in system:
                intent='退款退货' if any(w in raw for w in ['退货','邮费','运费']) else ('商品咨询' if '饮水机' in raw else ('订单' if '订单状态' in raw else '物流'))
                result={'intent':intent,'confidence':0.99}
            else:
                historical='最开始' in raw
                result={'question':'订单1001的物流及核实诉求是什么？' if historical else raw.split('以下补充')[0],
                    'reference_status':'resolved' if historical else 'clear',
                    'scope':'order_specific' if '1001' in raw or historical else 'general',
                    'order_id':'1001' if '1001' in raw or historical else None,
                    'order_source':'history' if historical else ('current' if '1001' in raw else 'none')}
            return AIMessage(content=json.dumps(result,ensure_ascii=False))
        if any(m.type=='tool' for m in messages) or '本轮知识证据": []' not in messages[-1].content:
            return AIMessage(content='',tool_calls=[{'id':'finish','name':'AgentDecision',
                'args':{'kind':'answer','tool_calls':[],'actions':[],'reason':'演示证据已足够'}}])
        return AIMessage(content='',tool_calls=[{'id':'c','name':'query_logistics','args':{'order_id':'1001'}}])
    async def astream(self,messages,**kwargs):
        from langchain_core.messages import AIMessageChunk
        text='订单1001为演示数据。您最早要求核实未到的物流，该诉求仍待核实；不能保证送达时间。'
        current=next((m.content for m in reversed(messages) if m.type=='human' and not m.additional_kwargs.get('context_materials')),'')
        if '最开始' in current and any('13800000000' in str(m.content) for m in messages):
            text+='最早留下的演示联系手机号是13800000000。'
        yield AIMessageChunk(content=text)

async def evaluate_dialogue(profile,mode,cases,direct=False):
    # Import installs app logging only; never enters lifespan or opens a port.
    import app.main
    from app.config import settings
    from app.context.contracts import ContextConfig
    from app.context.budget import calculate_budget
    from app.workflow.runtime import checkpoint_runtime
    from app.workflow.graph import build_workflow
    from app.workflow.service import WorkflowChatService
    from app.services.chat import DatabaseChatStore
    from app.db.session import SessionLocal
    from app.schemas.chat import ChatRequest
    from app.repositories.summaries import SummaryRepository
    from app.core.llm import get_chat_model
    from app.knowledge.retrieval import RetrievalResult
    from app.knowledge.query_understanding import QueryUnderstanding
    cfg=ContextConfig() if profile=='default' else ContextConfig(window=18000,output=2000,user_input=2000,steps=3,tool_result=1200,rerank_k=5)
    sig=signature(profile,mode,(settings.chat_model+(':direct-no-reuse' if direct else '')) if mode=='live' else 'LocalModel')
    path=ROOT/'reports'/f'eval-{mode}-{profile}-{sig[:12]}.json'
    if path.exists():
        report=json.loads(path.read_text(encoding='utf-8'))
        if report.get('assessment',{}).get('passed'): return path,report
        if report.get('error'): raise RuntimeError(f'Previous run failed; inspect {path}, use a changed implementation/new report before retry')
    else: report={'mode':mode,'profile':profile,'signature':sig,'config':asdict(cfg),'initial_budget':asdict(calculate_budget(cfg)),
        'user_id':'ch07-eval-'+uuid4().hex,'conversation_id':None,'turns':[]}
    model_factory=(lambda:get_chat_model(streaming=True,disable_thinking=True,max_retries=2,stream_usage=True,
        http_async_client=direct_client if direct else None)) if mode=='live' else LocalModel
    async def local_retrieve(query,budget):
        return RetrievalResult(QueryUnderstanding(normalized_query=query,search_query=query),[
            {'id':1,'question':'退货政策','answer':'商品完好且包装齐全，签收七天内可申请退货，地址以审核后提供的地址为准。','rerank_score':0.9}],True,'local_fixture')
    import httpx
    async with httpx.AsyncClient(trust_env=False,limits=httpx.Limits(max_keepalive_connections=0),timeout=60) as direct_client,checkpoint_runtime(path.with_suffix('.sqlite')) as saver:
        service=WorkflowChatService(build_workflow(checkpointer=saver),DatabaseChatStore(SessionLocal),model_factory=model_factory,
            context_config=cfg,retriever=local_retrieve if mode=='local' else None)
        try:
            for turn in cases['dialogues']['turns'][len(report['turns']):]:
                started=monotonic()
                req=ChatRequest(user_id=report['user_id'],conversation_id=report['conversation_id'],message=turn['question'])
                events=[event async for event in service.events(req)]
                report['conversation_id']=events[0][1]['conversation_id']
                text=''.join(data['delta'] for kind,data in events if kind=='delta')
                status=next((data for kind,data in reversed(events) if kind=='workflow_summary'),{})
                anchors,segments=await asyncio.to_thread(service.summary_repository.load,report['conversation_id'])
                snapshot=await service.graph.aget_state(service._config(report['conversation_id']))
                before=report.get('segments',[])
                current_segments=[asdict(s) for s in segments]
                row={'index':turn['index'],'answer':text,'stop_reason':status.get('stop_reason'),'usage':status.get('usage'),
                    'trace':status.get('node_trace'),'anchors':asdict(anchors),'seconds':monotonic()-started,
                    'checkpoint_count':len(snapshot.values.get('messages',[])),
                    'segments_immutable':current_segments[:len(before)]==before,
                    'passed':context_turn_completed(status,text),
                    'business_answered':status.get('stop_reason')=='answered'}
                report['turns'].append(row);save_report(path,report)
                print(f"{mode}/{profile} turn {row['index']}/22 stop={row['stop_reason']} seconds={row['seconds']:.2f}",flush=True)
                if not row['passed']: raise RuntimeError('dialogue turn did not converge')
                # User reply is complete. Wait only in evaluator, outside SSE,
                # before next synthetic user turn; proves no reply awaits jobs.
                if service.summary_worker.tasks: await asyncio.gather(*list(service.summary_worker.tasks.values()),return_exceptions=True)
                _,after_segments=await asyncio.to_thread(service.summary_repository.load,report['conversation_id'])
                after=[asdict(s) for s in after_segments]
                row['segments_immutable']=row['segments_immutable'] and after[:len(current_segments)]==current_segments
                report['segments']=after;save_report(path,report)
            anchors,segments=await asyncio.to_thread(service.summary_repository.load,report['conversation_id'])
            report['segments']=[asdict(s) for s in segments]
            report['final_anchors']=asdict(anchors)
            from app.context.layers import history_records
            from app.context.contracts import SummaryBatch
            from app.services.summarizer import validate_summary
            snapshot=await service.graph.aget_state(service._config(report['conversation_id']))
            records=history_records(snapshot.values['messages'],snapshot.values['message_db_ids'])
            checks=[]
            for segment in segments:
                source=tuple(r for r in records if segment.from_msg_id<=r.owner_msg_id<=segment.upto_msg_id)
                valid=bool(source) and min(r.owner_msg_id for r in source)==segment.from_msg_id and max(r.owner_msg_id for r in source)==segment.upto_msg_id
                try: validate_summary(segment.content,SummaryBatch(report['conversation_id'],0,segment.upto_msg_id,source,()))
                except ValueError: valid=False
                checks.append(valid)
            report['source_checks_passed']=all(checks) and [s.seq for s in segments]==list(range(1,len(segments)+1))
            report['context_cases']={c['id']:('dialogue_checked' if c.get('profile')==profile else 'separate_local_test' if c.get('failure') else 'other_profile') for c in cases['contexts']}
            report['assessment']=assess_dialogue(report,profile)
        except Exception as exc:
            report['error']={'type':type(exc).__name__,'message':str(exc)}
            save_report(path,report)
            raise
        finally: await service.summary_worker.aclose()
    save_report(path,report)
    return path,report

async def evaluate_summaries(cases):
    import app.main
    from app.core.llm import get_chat_model
    from app.config import settings
    from app.context.contracts import SummaryBatch,SummarySegment,HistoryRecord
    from app.services.summarizer import summary_messages,validate_summary
    from langchain_core.messages import HumanMessage
    sig=signature('summary','live',settings.chat_model)
    path=ROOT/'reports'/f'eval-live-summary-{sig[:12]}.json'
    report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'mode':'live','signature':sig,'results':{}}
    model=get_chat_model(disable_thinking=True,max_retries=0)
    for case in cases['summaries']:
        if case['id'] in report['results']: continue
        batch=SummaryBatch(0,0,len(case['source']),tuple(HistoryRecord(HumanMessage(content=text),i+1,'t') for i,text in enumerate(case['source'])),
            tuple(SummarySegment(i+1,1,1,text) for i,text in enumerate(case.get('background',[]))))
        response=await model.ainvoke(summary_messages(batch),max_tokens=600,response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}})
        content=json.loads(response.content)['summary']
        report['results'][case['id']]={'summary':content,**assess_summary(case,content),'usage':response.usage_metadata}
        try: validate_summary(content,batch)
        except ValueError as exc:
            report['results'][case['id']].update(passed=False,validation_error=str(exc))
        save_report(path,report)
        print(f"summary {case['id']} label_pass={report['results'][case['id']]['passed']}",flush=True)
    report['label_checks_passed']=all(item['passed'] for item in report['results'].values())
    save_report(path,report);return path,report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--live',action='store_true')
    mode.add_argument('--local-workflow',action='store_true')
    parser.add_argument('--profile',choices=['default','demo'],default='default')
    parser.add_argument('--section',choices=['dialogue','summary'],default='dialogue')
    parser.add_argument('--direct',action='store_true',help='Own direct HTTP client with no reused connections; evaluation process only')
    args=parser.parse_args();cases=load_cases();counts=inventory(cases)
    if not args.live and not args.local_workflow:
        path=ROOT/'reports'/'eval-offline.json';save_report(path,counts);print(json.dumps(counts));return 0
    if args.section=='summary':
        if not args.live: parser.error('summary semantic evaluation requires --live')
        path,report=asyncio.run(evaluate_summaries(cases));passed=report['label_checks_passed']
    else:
        path,report=asyncio.run(evaluate_dialogue(args.profile,'live' if args.live else 'local',cases,args.direct));passed=report['assessment']['passed']
    print(f'report={path} passed={passed}')
    return 0 if passed else 1

if __name__=='__main__': raise SystemExit(main())
