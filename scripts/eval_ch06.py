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


def evaluation_signature(model,threshold,files,configuration=None):
    from importlib.metadata import version
    versions={name:version(name) for name in ['langchain-core','langchain-openai','langgraph','pydantic']}
    digest=hashlib.sha256(json.dumps({'model':model,'confidence_threshold':threshold,'configuration':configuration or {},'versions':versions},sort_keys=True).encode())
    for path in sorted(files):
        digest.update(str(path).encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def query_signature(queries):
    return hashlib.sha256(json.dumps(queries,ensure_ascii=False).encode()).hexdigest()


def check_expansion_quality(case,queries,review):
    contract=(bool(queries) and queries[0]==case['question'] and len(set(queries))==len(queries)
              and 1<=len(queries)<=3 and (len(queries)>1 if case['must_expand'] else queries==[case['question']]))
    if not case['must_expand']:
        return {'passed':contract,'contract_valid':contract,'review_status':'not_required','queries':queries}
    valid=isinstance(review,dict) and review.get('query_signature')==query_signature(queries)
    approved=valid and review.get('facets_ok') is True and review.get('facts_preserved') is True and bool(review.get('reason'))
    return {'passed':bool(contract and approved),'contract_valid':contract,
            'review_status':'approved' if approved else ('rejected' if valid else 'pending'),'queries':queries,
            'query_signature':query_signature(queries),'review':review if valid else None}


async def evaluate_assessment(model,case):
    from app.workflow.agent import build_agent_graph
    from app.workflow.after_sales import refund_offer_defaults
    from app.workflow.contracts import AgentDependencies,TurnContext
    from app.workflow.budget import TurnBudget
    from app.tools.business import build_tools
    from app.workflow.bare_loop import tool_error
    budget=TurnBudget()
    calls=[]
    class EvidenceOnlyExecutor:
        async def execute(self,call,before_attempt=None):
            if before_attempt: before_attempt()
            calls.append(call['name'])
            return tool_error(call,'evaluation_contains_all_order_and_policy_facts')
    # Advertise the same production tools; prevent unexpected requests from
    # consulting another source and hiding a bad decision behind extra evidence.
    tools=build_tools(lambda:None,1)
    context=TurnContext(AgentDependencies(model,tools,EvidenceOnlyExecutor()),budget)
    state={'messages':[HumanMessage(content='订单1001能否申请退货？')],'turn_id':case['id'],
        'question':'订单1001能否申请退货？','intent':case['intent'],'order_id':'1001',
        'order_snapshot':{'mock':True,'order_id':'1001',**case['facts']},'gate_allowed':True,
        'evidence':[{'n':1,'chunk_id':1,'question':'退货政策','answer':case['policy'],'section_path':'演示政策/退货'}],
        'citations':[{'n':1,'chunk_id':1,'section_path':'演示政策/退货'}]}
    result=await build_agent_graph().ainvoke(state,context=context)
    assessment=result.get('refund_assessment',{})
    references=assessment.get('evidence_numbers',[])
    offer=refund_offer_defaults(result) is not None and 'refund' in result.get('actions',[])
    expected_offer=case['expected_status']=='eligible'
    valid_refs=set(references).issubset({1}) and (bool(references) if expected_offer else True)
    passed=(assessment.get('status')==case['expected_status'] and valid_refs and offer==expected_offer
            and (expected_offer or 'refund' not in result.get('actions',[])) and not calls
            and result.get('stop_reason') in {'answered','clarify'})
    return {'passed':passed,'actual':assessment,'expected':case['expected_status'],'offer_allowed':offer,
            'references_valid':valid_refs,'unexpected_tools':calls,'actions':result.get('actions',[]),'answer':result.get('final_text'),
            'production_trace':result.get('node_trace',[])}


async def live_eval(cases,report,path,section,retry_failed,expansion_reviews=None):
    from app.core.llm import get_chat_model
    from app.workflow.intent import classify_with_escalation,detect_smalltalk
    from app.workflow.resolution import resolve_question
    from app.workflow.expansion import expand_queries
    from app.workflow.order_context import demo_order
    from app.workflow.routing_contracts import Resolution,RefundAssessment
    from app.workflow.budget import TurnBudget
    from app.workflow.model import BudgetedModel
    model=get_chat_model(disable_thinking=True,max_retries=0)
    from app.config import settings
    small_model=get_chat_model(model_name=settings.intent_small_model,disable_thinking=True,max_retries=0) if settings.intent_small_model else None
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
                run=await classify_with_escalation(model,small_model,case['query'],TurnBudget(),settings.intent_confidence_threshold)
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
                    intent=await classify_with_escalation(model,small_model,resolution.question,budget,settings.intent_confidence_threshold)
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
            cached=results.get(case['id'])
            review=(expansion_reviews or {}).get(case['id']) or (cached or {}).get('review')
            if cached and 'queries' in cached and not retry_failed:
                results[case['id']]={**cached,**check_expansion_quality(case,cached['queries'],review)}
                save_report(path,report)
                continue
            async def check(case=case):
                from app.workflow.order_context import trusted_order_ids
                ids=trusted_order_ids(case['question'],[])['current']
                snapshot=demo_order(next(iter(ids))) if len(ids)==1 else {}
                queries=await expand_queries(model,case['question'],snapshot,case['intent'],TurnBudget())
                return {**check_expansion_quality(case,queries,review),'expected_facets':case['facets'],'order_snapshot':snapshot}
            await execute(case['id'],check)
    if section in {'all','assessment'}:
        for case in cases['assessment']:
            async def check(case=case):
                return await evaluate_assessment(model,case)
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
    pending_review=any(row.get('review_status')=='pending' for key,row in results.items() if key.startswith('X'))
    report.update(quality_status='passed' if passed else ('pending_review' if complete and pending_review else ('failed' if complete else 'partial')),complete=complete)
    save_report(path,report)
    print(f'Quality status: {report["quality_status"]}; intent accuracy: {accuracy}')
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',help='Send demonstration evaluation data to configured chat upstream after authorization')
    parser.add_argument('--section',choices=['all','intent','dialogues','expansion','assessment'],default='all')
    parser.add_argument('--retry-failed',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--expansion-review',type=Path,help='Human review JSON keyed by case id with query_signature,facets_ok,facts_preserved,reason')
    args=parser.parse_args()
    root=Path('data/ch06'); cases=load_cases(root); counts=inventory(cases)
    if not args.live:
        report={'mode':'offline inventory','inventory':counts,'quality_status':'not_run','complete':False,'reason':'No model invoked; dataset validation is not accuracy evaluation.'}
        save_report(args.output or root/'reports/eval-offline.json',report)
        print(json.dumps(report,ensure_ascii=False)); return
    from app.config import settings
    files=[*root.glob('*.json'),*root.glob('*.jsonl'),*Path('app/workflow').glob('*.py'),
           Path('app/core/workflow_prompts.py'),Path('app/core/llm.py'),Path('app/tools/business.py'),
           *Path('app/knowledge').glob('*.py'),Path('app/services/refunds.py'),Path('app/config.py'),Path(__file__)]
    signature=evaluation_signature(settings.chat_model,settings.intent_confidence_threshold,files,
        {'upstream':settings.chat_base_url,'small_model':settings.intent_small_model})
    path=args.output or root/'reports/eval-live.json'
    report=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'mode':'live','model':settings.chat_model,'signature':signature,'inventory':counts,'results':{}}
    if report.get('signature')!=signature: raise SystemExit('Inputs/model changed; choose a new --output report, do not overwrite old evidence.')
    reviews=json.loads(args.expansion_review.read_text(encoding='utf-8')) if args.expansion_review else {}
    if not asyncio.run(live_eval(cases,report,path,args.section,args.retry_failed,reviews)): raise SystemExit(1)


if __name__=='__main__': main()
