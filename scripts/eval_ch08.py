"""Local deterministic checks and explicitly authorized synthetic live evaluation."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'data/ch08/reports'


def save(report,name):
    REPORTS.mkdir(parents=True,exist_ok=True)
    (REPORTS/name).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


def run_local():
    migration=subprocess.run([sys.executable,'-m','scripts.migrate_ch08','--check'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if migration.returncode:
        return {'mode':'local','status':'failed','migration_check':migration.stdout+ migration.stderr,'returncode':migration.returncode}
    suites=['tests/test_ch08_registry.py','tests/test_ch08_audit.py','tests/test_ch08_executor.py',
        'tests/test_ch08_mcp.py','tests/test_ch08_ticket_confirmation.py','tests/test_ch08_ticket_workflow.py',
        'tests/test_ch08_confirmation_api.py','tests/test_ch08_context_budget.py','tests/test_ch08_integration.py',
        'tests/test_ch08_history_redaction.py','tests/test_ch08_live_eval.py']
    tests=subprocess.run([sys.executable,'-m','pytest','-q',*suites],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    return {'mode':'local','model_mode':'offline substitutes + actual local MCP subprocesses + test MySQL',
        'status':'passed' if tests.returncode==0 else 'failed','migration_check':migration.stdout.strip(),
        'pytest_returncode':tests.returncode,'pytest_summary':tests.stdout.splitlines()[-1] if tests.stdout else '',
        'pytest_output':(tests.stdout+'\n'+tests.stderr)[-12000:],
        'cases':[json.loads(line) for line in (ROOT/'data/ch08/tool_cases.jsonl').read_text(encoding='utf-8-sig').splitlines() if line],
        'timestamp_utc':datetime.now(timezone.utc).isoformat()}


def live_tool_cases():
    """Fixed synthetic examples only; never loads conversation or customer records."""
    return [
        {'id':'L01','question':'订单1001的物流轨迹到哪里了？','expected_tool':'query_logistics','expected_server':'logistics'},
        {'id':'L02','question':'订单1001对应商品现在还在保修期内吗？','expected_tool':'query_warranty','expected_server':'after_sales'},
    ]


def grade_live_tool_call(message,expected_tool):
    calls=getattr(message,'tool_calls',[]) or []
    correct=len(calls)==1 and calls[0].get('name')==expected_tool and bool(calls[0].get('id'))
    return {'passed':correct,'tool_name':calls[0].get('name') if len(calls)==1 else None,
        'tool_call_id':calls[0].get('id') if len(calls)==1 else None}


def live_ticket_cases():
    return [
        {'id':'T-L01','question':'帮我建个工单','expected_action':'clarify'},
        {'id':'T-L02','question':'订单1001包裹破损，请帮我建一个售后工单。',
         'expected_action':'preview','required_description':'订单1001包裹破损','expected_type':'售后'},
    ]


def grade_live_ticket(message,case):
    calls=getattr(message,'tool_calls',[]) or []
    if case['expected_action']=='clarify':
        content=getattr(message,'content','')
        text=content if isinstance(content,str) else str(content)
        asks_for_detail=any(word in text for word in ('描述','具体问题','什么问题','详情'))
        return {'passed':not calls and asks_for_detail,'action':'clarify' if not calls and asks_for_detail else 'unexpected_tool_or_answer',
            'tool_call_id':None}
    valid=(len(calls)==1 and calls[0].get('name')=='create_ticket' and bool(calls[0].get('id')))
    args=calls[0].get('args',{}) if valid else {}
    grounded=case['required_description'] in str(args.get('description',''))
    correct_type=args.get('ticket_type')==case['expected_type']
    return {'passed':valid and grounded and correct_type,'action':'preview' if valid else 'no_preview',
        'tool_call_id':calls[0].get('id') if valid else None,'description_grounded':grounded,
        'ticket_type_correct':correct_type,'execution':'not_executed_no_ticket_created'}


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


async def _start_mcp_servers():
    processes={};connections={}
    for name,module in [('logistics','mcp_servers.logistics'),('after_sales','mcp_servers.after_sales')]:
        port=_free_port()
        env=os.environ.copy()
        for secret_name in ('CHAT_API_KEY','EMBED_API_KEY','RERANK_API_KEY','DATABASE_URL','MYSQL_PWD'):
            env.pop(secret_name,None)
        env.update(MCP_PORT=str(port),PYTHONPATH=str(ROOT),PYTHONIOENCODING='utf-8')
        process=subprocess.Popen([sys.executable,'-m',module],cwd=ROOT,env=env,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        processes[name]=process
        ready=False
        for _ in range(100):
            if process.poll() is not None: break
            try:
                reader,writer=await asyncio.open_connection('127.0.0.1',port)
                writer.close();await writer.wait_closed();ready=True;break
            except OSError: await asyncio.sleep(.1)
        if not ready:
            for item in processes.values():
                if item.poll() is None: item.terminate()
            raise RuntimeError(f'本地 MCP {name} 演示服务未能启动')
        connections[name]={'url':f'http://127.0.0.1:{port}/mcp','transport':'streamable_http'}
    return processes,connections


async def _run_live_deepseek():
    from langchain_core.messages import SystemMessage,HumanMessage
    from app.config import settings
    from app.core.llm import get_chat_model
    from app.db.session import SessionLocal
    from app.tooling.contracts import CallContext
    from app.tooling.policy import PolicyStore
    from app.tooling.registry import ToolRegistry
    from app.tooling.builtins import bind_builtins
    from app.tooling.mcp_client import MCPDiscovery
    from app.services.tool_executor import ToolExecutor

    if not settings.chat_api_key:
        raise RuntimeError('CHAT_API_KEY 未配置')
    processes,connections=await _start_mcp_servers()
    discovery=None
    audits=[]
    class AuditSink:
        def write(self,record): audits.append(record)
    try:
        policy=PolicyStore(ROOT/settings.tool_policy_path)
        registry=ToolRegistry(policy,ROOT/settings.tool_plugins_path)
        registry.register_builtin(bind_builtins)
        context=CallContext(conversation_id=880001,user_id='ch08-synthetic-eval',turn_id='live-eval',
                            session_factory=SessionLocal)
        discovery=MCPDiscovery(connections)
        discovered=await discovery.discover()
        snapshot=registry.snapshot(context,discovered)
        model=get_chat_model(streaming=False,disable_thinking=True,max_retries=0)
        model_with_tools=model.bind_tools(snapshot.model_tools())
        executor=ToolExecutor(snapshot,context=context,policy=policy,audits=AuditSink())
        output=[]
        system=SystemMessage(content='你是工具选择验收助手。必须根据用户问题调用且只调用一个最合适的查询工具；物流问题用query_logistics，在保修问题用query_warranty。订单号按字符串传入。调用后只依据工具结果简短回答，并说明这是演示mock数据。')
        for case in live_tool_cases():
            first=await asyncio.wait_for(model_with_tools.ainvoke([system,HumanMessage(content=case['question'])]),timeout=75)
            grade=grade_live_tool_call(first,case['expected_tool'])
            row={'id':case['id'],'provider':'DeepSeek','model':settings.chat_model,
                'question':case['question'],'expected_tool':case['expected_tool'],
                'actual_tool':grade['tool_name'],'tool_call_id':grade['tool_call_id'],
                'tool_call_passed':grade['passed'],'tool_source':None,'mcp_server':None,
                'tool_result':None,'final_answer':None,'status':'failed'}
            if grade['passed']:
                definition=snapshot.get(grade['tool_name'])
                row.update(tool_source=definition.source,mcp_server=definition.mcp_server)
                tool_message=await asyncio.wait_for(executor.execute(first.tool_calls[0]),timeout=20)
                row['tool_result']=json.loads(tool_message.content)
                final=await asyncio.wait_for(model.ainvoke([system,HumanMessage(content=case['question']),first,tool_message]),timeout=75)
                row['final_answer']=final.content if isinstance(final.content,str) else str(final.content)
                row['status']='passed' if definition.mcp_server==case['expected_server'] and bool(row['final_answer'].strip()) else 'failed'
            output.append(row)
        ticket_definition=snapshot.get('create_ticket')
        ticket_model=model.bind_tools([ticket_definition.tool])
        ticket_system=SystemMessage(content='你是工单申请助手。信息不完整时先向客户澄清，严禁补造问题描述；只有用户明确要求且提供具体问题描述时，才调用create_ticket生成待用户确认的预览。本验收不会执行提交。description必须摘录客户原话，工单类型从用户表达判断。')
        ticket_output=[]
        for case in live_ticket_cases():
            message=await asyncio.wait_for(ticket_model.ainvoke([ticket_system,HumanMessage(content=case['question'])]),timeout=75)
            grade=grade_live_ticket(message,case)
            ticket_output.append({'id':case['id'],'provider':'DeepSeek','model':settings.chat_model,
                'question':case['question'],'expected_action':case['expected_action'],**grade,
                'status':'passed' if grade['passed'] else 'failed'})
        return {'provider':'DeepSeek','model':settings.chat_model,'cases':output,
                'ticket_cases':ticket_output,
                'audit_records':audits,'mcp_servers_discovered':sorted(connections),
                'status':'passed' if all(row['status']=='passed' for row in [*output,*ticket_output]) else 'failed'}
    finally:
        for process in processes.values():
            if process.poll() is None: process.terminate()
            try: await asyncio.to_thread(process.wait,timeout=5)
            except subprocess.TimeoutExpired: process.kill();await asyncio.to_thread(process.wait,timeout=5)


async def _run_live_siliconflow():
    import httpx
    from app.config import settings
    key=settings.rerank_api_key or settings.embed_api_key
    if not key: raise RuntimeError('RERANK_API_KEY/EMBED_API_KEY 未配置')
    query='订单1001的物流轨迹到哪里了？'
    documents=[
        'query_logistics：按订单号查询物流状态、当前位置和轨迹。',
        'query_warranty：查询指定订单商品是否在保及保修依据。',
        'query_order：查询订单状态、商品及金额信息。',
        'create_ticket：在用户明确要求并确认后创建人工工单。',
    ]
    payload={'model':settings.rerank_model,'query':query,'documents':documents,
             'top_n':len(documents),'return_documents':False}
    async with httpx.AsyncClient(timeout=45,trust_env=True) as client:
        for attempt in range(3):
            try:
                response=await client.post(settings.rerank_base_url.rstrip('/')+'/rerank',
                    headers={'Authorization':f'Bearer {key}'},json=payload)
                break
            except httpx.TransportError:
                if attempt==2: raise
                await asyncio.sleep(attempt+1)
    response.raise_for_status()
    data=response.json();results=data.get('results',[])
    ranked=[{'tool_name':documents[int(item['index'])].split('：',1)[0],
             'score':item.get('relevance_score')} for item in results]
    return {'provider':'SiliconFlow','model':settings.rerank_model,'query':query,'ranking':ranked,
            'transport_retries':attempt,
            'status':'passed' if ranked and ranked[0]['tool_name']=='query_logistics' else 'failed'}


async def run_live():
    report={'mode':'live','data_scope':'仅固定合成问题/订单1001演示mock结果；未加载客户会话或消息',
        'side_effects':'不写MySQL、不创建工单；审计仅保存在进程内。',
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'status':'failed'}
    try:
        report['deepseek']=await _run_live_deepseek()
    except Exception as exc:
        report['deepseek']={'status':'failed','error_type':type(exc).__name__}
    try:
        report['siliconflow']=await _run_live_siliconflow()
    except Exception as exc:
        report['siliconflow']={'status':'failed','error_type':type(exc).__name__}
    report['status']='passed' if report.get('deepseek',{}).get('status')=='passed' and report.get('siliconflow',{}).get('status')=='passed' else 'failed'
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    return report


def main():
    parser=argparse.ArgumentParser(); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--local',action='store_true'); mode.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.live:
        report=asyncio.run(run_live())
        save(report,'live.json');print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['status']=='passed' else 2
    report=run_local();save(report,'local.json');print(json.dumps({k:v for k,v in report.items() if k!='pytest_output'},ensure_ascii=False,indent=2));return report.get('returncode',report.get('pytest_returncode',0))


if __name__=='__main__':raise SystemExit(main())
