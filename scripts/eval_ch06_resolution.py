"""Small labelled prompt gate; only repository demonstration dialogue is sent."""
import asyncio
import json
from pathlib import Path
from langchain_core.messages import HumanMessage,AIMessage
from app.core.llm import get_chat_model
from app.workflow.resolution import resolve_question
from app.workflow.budget import TurnBudget


async def main():
    model=get_chat_model(disable_thinking=True,max_retries=0)
    results=[]
    for dialogue in json.loads(Path('data/ch06/dialogues.json').read_text(encoding='utf-8')):
        history=[]
        target=next((t for t in dialogue['turns'] if t.get('expected_reference_status') in {'resolved','ambiguous'}),dialogue['turns'][0])
        for turn in dialogue['turns']:
            if turn is target: break
            history.append((HumanMessage if turn['role']=='user' else AIMessage)(content=turn['content']))
        result=await resolve_question(model,target['content'],history,TurnBudget())
        passed=(result.reference_status==target['expected_reference_status'] and result.order_id==target['expected_order_id']
            and result.scope==target['expected_scope'] and all(t in result.question for t in target['required_terms'])
            and (not target['unchanged'] or result.question==target['content']))
        results.append({'id':dialogue['id'],'passed':passed,'actual':result.model_dump()})
    output=Path('reports/ch06-resolution-smoke.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{sum(r["passed"] for r in results)}/{len(results)} labelled resolution cases passed')
    if not all(r['passed'] for r in results): raise SystemExit(1)


if __name__=='__main__': asyncio.run(main())
