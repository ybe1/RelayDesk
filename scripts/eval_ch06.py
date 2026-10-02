"""Labelled routing evaluation. Offline validates data, never claims model quality."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from collections import Counter
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage


def load_cases(root):
    def lines(name): return [json.loads(line) for line in (root/name).read_text(encoding='utf-8').splitlines() if line.strip()]
    return {'intent':lines('intent_cases.jsonl'),'dialogues':json.loads((root/'dialogues.json').read_text(encoding='utf-8')),
            'expansion':lines('expansion_cases.jsonl'),'assessment':json.loads((root/'refund_assessment_cases.json').read_text(encoding='utf-8'))}


def inventory(cases):
    labels=Counter(case['expected_intent'] for case in cases['intent'])
    expected={'物流','订单','商品咨询','退款退货','售后','投诉','闲聊','其他'}
    if set(labels)!=expected or min(labels.values())<3 or len(cases['intent'])<40: raise ValueError('intent labels incomplete')
    turns=[sum(turn['role']=='user' for turn in d['turns']) for d in cases['dialogues']]
    if len(turns)<6 or min(turns)<4: raise ValueError('dialogues incomplete')
    if len(cases['expansion'])<8 or len(cases['assessment'])<3: raise ValueError('core samples incomplete')
    return {'intent':len(cases['intent']),'dialogues':len(turns),'dialogue_turns':sum(turns),
            'expansion':len(cases['expansion']),'assessment':len(cases['assessment'])}


def should_run(key,results,retry_failed):
    return key not in results or (retry_failed and not results[key].get('passed'))


def save_report(path,report):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def required_term_present(term,text):
    alternatives={'物流':['物流','快递','配送','包裹','运输']}
    return any(value in text for value in alternatives.get(term,[term]))


async def live_eval(cases,report,path,section,retry_failed):
    from app.core.llm import get_chat_model
    from app.workflow.intent import classify_with_escalation,detect_smalltalk
    from app.workflow.resolution import resolve_question
    from app.workflow.expansion import expand_queries
    from app.workflow.order_context import demo_order
    from app.workflow.routing_contracts import Resolution,RefundAssessment
    from app.workflow.budget import TurnBudget
    from app.workflow.model import BudgetedModel
    model=get_chat_model(disable_thinking=True,max_retries=0)
    results=report.setdefault('results',{})
    async def execute(key,operation):
        if not should_run(key,results,retry_failed): return results[key]
        try: outcome=await operation()
        except Exception as exc:
            results[key]={'passed':False,'error_type':type(exc).__name__}
            report['quality_status']='blocked'
            save_report(path,report)
            print(f'{key}: blocked ({type(exc).__name__}); no further requests')
            raise
        results[key]=outcome; save_report(path,report)
        print(f'{key}: {"PASS" if outcome["passed"] else "FAIL"}',flush=True)
        return outcome
    if section in {'all','intent'}:
        for case in cases['intent']:
            async def check(case=case):
                run=await classify_with_escalation(model,None,case['query'],TurnBudget())
                return {'passed':run.decision.intent==case['expected_intent'],'json_valid':True,
                        'expected':case['expected_intent'],'actual':run.decision.model_dump(),'bucket':case['bucket']}
            await execute(case['id'],check)
    if section in {'all','dialogues'}:
        for dialogue in cases['dialogues']:
            history=[]
            for index,turn in enumerate(dialogue['turns']):
                if turn['role']=='assistant': history.append(AIMessage(content=turn['content'])); continue
                async def check(turn=turn,history=list(history)):
                    budget=TurnBudget()
                    if detect_smalltalk(turn['content']):
                        resolution=Resolution(question=turn['content'],reference_status='clear',scope='general',order_source='none')
                    else: resolution=await resolve_question(model,turn['content'],history,budget)
                    intent=await classify_with_escalation(model,None,resolution.question,budget)
                    checks={'intent':intent.decision.intent==turn['expected_intent'],
                        'scope':resolution.scope==turn['expected_scope'],
                        'reference':resolution.reference_status==turn['expected_reference_status'],
                        'order':resolution.order_id==turn['expected_order_id'],
                        'terms':all(required_term_present(term,resolution.question) for term in turn['required_terms']),
                        'unchanged':not turn['unchanged'] or resolution.question==turn['content']}
                    return {'passed':all(checks.values()),'checks':checks,'resolution':resolution.model_dump(),
                            'intent':intent.decision.model_dump(),'expected':turn}
                await execute(f'{dialogue["id"]}:{index}',check)
                history.append(HumanMessage(content=turn['content']))
    if section in {'all','expansion'}:
        for case in cases['expansion']:
            async def check(case=case):
                queries=await expand_queries(model,case['question'],demo_order('1001'),case['intent'],TurnBudget())
                valid=(queries[0]==case['question'] and len(set(queries))==len(queries) and 1<=len(queries)<=3)
                valid=valid and (len(queries)>1 if case['must_expand'] else queries==[case['question']])
                return {'passed':valid,'queries':queries,'expected_facets':case['facets'],
                        'note':'Contract/count verified; inspect distinct retrieval facets in report.'}
            await execute(case['id'],check)
    if section in {'all','assessment'}:
        for case in cases['assessment']:
            async def check(case=case):
                response=await BudgetedModel(model,TurnBudget(),384).ainvoke([
                    SystemMessage(content='仅按订单事实与政策判断退款资格。只输出JSON：{"status":"eligible/ineligible/needs_review","reason":"原因","evidence_numbers":[1]}。未知关键条件归needs_review，不猜商品或包装完好；不得承诺到账时间。'),
                    HumanMessage(content=json.dumps({'intent':case['intent'],'facts':case['facts'],'evidence':[{'n':1,'answer':case['policy']}]},ensure_ascii=False))],
                    response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}})
                assessment=RefundAssessment.model_validate(json.loads(response.content))
                return {'passed':assessment.status==case['expected_status'],'actual':assessment.model_dump(),'expected':case['expected_status']}
            await execute(case['id'],check)
    intent_rows=[results[case['id']] for case in cases['intent'] if case['id'] in results]
    accuracy=sum(row['passed'] for row in intent_rows)/len(intent_rows) if intent_rows else None
    report['intent_accuracy']=accuracy
    report['intent_json_compliance']=sum(row.get('json_valid',False) for row in intent_rows)/len(intent_rows) if intent_rows else None
    expected_keys=[case['id'] for group in ['intent','expansion','assessment'] for case in cases[group]]
    expected_keys += [f'{d["id"]}:{i}' for d in cases['dialogues'] for i,t in enumerate(d['turns']) if t['role']=='user']
    complete=all(key in results and 'error_type' not in results[key] for key in expected_keys)
    other_keys=[key for key in expected_keys if not key.startswith('I')]
    passed=complete and accuracy is not None and accuracy>=0.95 and report['intent_json_compliance']==1 and all(results[key]['passed'] for key in other_keys)
    report.update(quality_status='passed' if passed else ('failed' if complete else 'partial'),complete=complete)
    save_report(path,report)
    print(f'Quality status: {report["quality_status"]}; intent accuracy: {accuracy}')
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Send demonstration evaluation data to configured chat upstream after authorization')
    parser.add_argument('--section',choices=['all','intent','dialogues','expansion','assessment'],default='all')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=Path('data/ch06'); cases=load_cases(root); counts=inventory(cases)
    if not args.live:
        report={'mode':'offline inventory','inventory':counts,'quality_status':'not_run','complete':False,'reason':'No model invoked; dataset validation is not accuracy evaluation.'}
        save_report(args.output or root/'reports/eval-offline.json',report)
        print(json.dumps(report,ensure_ascii=False)); return
    from app.config import settings
    files=[*root.glob('*.json'),*root.glob('*.jsonl'),Path('app/core/workflow_prompts.py'),Path('app/workflow/expansion.py')]
    signature=hashlib.sha256(settings.chat_model.encode()+b''.join(path.read_bytes() for path in sorted(files))).hexdigest()
    path=args.output or root/'reports/eval-live.json'
    report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'mode':'live','model':settings.chat_model,'signature':signature,'inventory':counts,'results':{}}
    if report.get('signature')!=signature: raise SystemExit('Inputs/model changed; choose a new --output report, do not overwrite old evidence.')
    if not asyncio.run(live_eval(cases,report,path,args.section,args.retry_failed)): raise SystemExit(1)


if __name__=='__main__': main()
