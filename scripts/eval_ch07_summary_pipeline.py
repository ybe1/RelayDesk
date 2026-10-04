"""Real asynchronous summary/recall probe on explicitly seeded synthetic history."""
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from uuid import uuid4
import httpx
from langchain_core.messages import HumanMessage, AIMessage
from app import main  # installs logging; does not start a web server
from app.context.contracts import ContextConfig
from app.core.llm import get_chat_model
from app.db.session import SessionLocal
from app.schemas.chat import ChatRequest
from app.services.chat import DatabaseChatStore
from app.workflow.runtime import checkpoint_runtime
from app.workflow.graph import build_workflow
from app.workflow.service import WorkflowChatService

async def run():
    root=Path('data/ch07/reports'); root.mkdir(parents=True,exist_ok=True)
    report={'scope':'seeded synthetic history, real background summary and real recall; not 22 live dialogue rounds'}
    user='ch07-summary-probe-'+uuid4().hex
    cfg=ContextConfig(window=18000,output=2000,user_input=2000,steps=3,tool_result=1200,rerank_k=5)
    store=DatabaseChatStore(SessionLocal); cid=store.start_or_load(user,None)
    report['conversation_id']=cid
    async with httpx.AsyncClient(trust_env=False,limits=httpx.Limits(max_keepalive_connections=0),timeout=60) as client:
        factory=lambda:get_chat_model(streaming=False,disable_thinking=True,max_retries=0,http_async_client=client)
        async with checkpoint_runtime(root/(user+'.sqlite')) as saver:
            graph=build_workflow(checkpointer=saver)
            service=WorkflowChatService(graph,store,model_factory=factory,context_config=cfg)
            try:
                fixture=json.loads(Path('data/ch07/dialogues.json').read_text(encoding='utf-8-sig'))['turns']
                messages=[]; ids={}
                for n,item in enumerate(fixture[:20]):
                    for m in [HumanMessage(content=item['question'],id=f'seed{n}:user'),
                              AIMessage(content='这是合成历史中的确认记录：您的核实诉求仍待确认，不承诺送达或到账时间。',id=f'seed{n}:final')]:
                        ids[m.id]=service._save(cid,f'seed{n}',m); messages.append(m)
                await graph.aupdate_state(service._config(cid),{'messages':messages,'message_db_ids':ids},as_node='log')
                started=monotonic()
                events=[e async for e in service.events(ChatRequest(user_id=user,conversation_id=cid,message='你好'))]
                report['reply_seconds']=monotonic()-started
                report['reply_done']=events[-1][0]=='done'
                report['anchors_at_reply']=asdict(service.summary_repository.load(cid)[0])
                report['summary_pending_at_reply']=bool(service.summary_worker.tasks)
                tasks=list(service.summary_worker.tasks.values())
                if tasks: await asyncio.wait_for(asyncio.gather(*tasks),150)
                anchors,segments=service.summary_repository.load(cid)
                report['summary_finished_seconds']=monotonic()-started
                report['anchors_after_summary']=asdict(anchors)
                report['segments']=[asdict(s) for s in segments]
                summary='\n'.join(s.content for s in segments)
                report['early_facts_preserved']=all(t in summary for t in ['1001','13800000000','核实'])
                if not segments: raise RuntimeError('background summary produced no segments')
                question='最开始那个订单号是什么？我留下的联系手机号是什么？当时想让你帮我做什么？'
                events=[e async for e in service.events(ChatRequest(user_id=user,conversation_id=cid,message=question))]
                answer=''.join(d['delta'] for k,d in events if k=='delta')
                report['recall_answer']=answer
                report['recall_passed']=all(t in answer for t in ['1001','13800000000','核实'])
                report['segments_immutable']=segments==service.summary_repository.load(cid)[1]
                report['passed']=all(report[k] for k in ['reply_done','summary_pending_at_reply','early_facts_preserved','recall_passed','segments_immutable'])
            except Exception as exc:
                report['passed']=False; report['error']=type(exc).__name__+': '+str(exc)
            finally:
                await service.summary_worker.aclose()
                path=root/'summary-pipeline-live.json'
                path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': asyncio.run(run())
