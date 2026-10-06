"""Headless acceptance for Ch09 review and observability pages with intercepted APIs."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


def verify(base_url):
    from playwright.sync_api import expect, sync_playwright
    root=Path('app/static'); decisions=[]
    items=[{'id':1,'normalized_question':'平台保修期多久？','ai_suggested_answer':'演示保修期为一年。','occurrence_count':3,
            'review_status':'待审','approved_answer':None,'created_at':'2026-10-06T00:00:00','updated_at':'2026-10-06T00:00:00'},
           {'id':2,'normalized_question':'不存在的演示问题？','ai_suggested_answer':None,'occurrence_count':1,
            'review_status':'待审','approved_answer':None,'created_at':'2026-10-06T00:03:00','updated_at':'2026-10-06T00:03:00'}]
    raw=[
      {'id':1,'raw_question':'这个保多久','source':'retrieval_low_conf','reason':'历史','retrieved_chunks':None,'conversation_id':1,'created_at':'2026-10-06T00:00:00'},
      {'id':2,'raw_question':'保修呢','source':'self_check','reason':'零命中','retrieved_chunks':[],'conversation_id':1,'created_at':'2026-10-06T00:01:00'},
      {'id':3,'raw_question':'保修政策','source':'user_feedback','reason':'证据弱','retrieved_chunks':[{'section_path':'售后/保修','rerank_score':0.42,'answer':'旧证据'}],'conversation_id':1,'created_at':'2026-10-06T00:02:00'}]
    def handle(route):
        req=route.request; path=urlparse(req.url).path
        if path in ('/review','/observability'):
            name='review.html' if path=='/review' else 'observability.html';return route.fulfill(content_type='text/html',body=(root/name).read_text(encoding='utf-8'))
        if path.startswith('/static/'):
            file=root/path.split('/')[-1];return route.fulfill(content_type='text/css' if file.suffix=='.css' else 'text/javascript',body=file.read_text(encoding='utf-8'))
        if path=='/api/review-queue' and req.method=='GET': return route.fulfill(json={'items':items,'total':len(items),'page':1,'page_size':100})
        if path.startswith('/api/review-queue/') and req.method=='GET':
            row=next(item for item in items if str(item['id'])==path.rsplit('/',1)[-1]);return route.fulfill(json={**row,'raw_questions':raw if row['id']==1 else []})
        if path.endswith('/decision'):
            body=req.post_data_json;decisions.append(body);rid=int(path.split('/')[-2]);items[:]=[item for item in items if item['id']!=rid];return route.fulfill(json={'status':'ok'})
        if path=='/api/observability/intent-usage': return route.fulfill(status=503,json={'detail':'Langfuse usage unavailable'})
        if path=='/api/eval-runs': return route.fulfill(json={'items':[{'id':2,'triggered_by':'手动','dataset_size':300,'created_at':'2026-10-06T02:00:00','metrics':{'strategies':{'hybrid_rerank':{'recall_at_5':.9,'mrr':.8}},'faithfulness':.95,'refusal_rate':1,'failed':0,'denominators':{'faithfulness':220},'dataset_hash':'d2','corpus_hash':'c2'}},{'id':1,'triggered_by':'定时','dataset_size':300,'created_at':'2026-09-29T02:00:00','metrics':{'strategies':{'hybrid_rerank':{'recall_at_5':.8,'mrr':.7}},'faithfulness':.9,'refusal_rate':.95,'failed':1,'denominators':{'faithfulness':218},'dataset_hash':'d1','corpus_hash':'c1'}}]})
        if path.startswith('/api/jobs/'): return route.fulfill(status=202,json={'state':'running'})
        return route.fulfill(status=404,body='not found')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True);context=browser.new_context();context.route('**/*',handle);page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(base_url.rstrip('/')+'/review');expect(page.get_by_text('平台保修期多久？')).to_be_visible();page.get_by_role('button',name='查看原话与证据').first.click();expect(page.get_by_text('历史记录 / 当轮未检索')).to_be_visible();expect(page.get_by_text('检索完成但零命中')).to_be_visible();expect(page.get_by_text('当轮 Top 3 证据')).to_be_visible();page.get_by_role('button',name='通过',exact=True).first.click();expect(page.locator('#approve-dialog')).to_be_visible();page.locator('#approved-answer').fill('核准答案');page.get_by_role('button',name='确认通过并入库').click();expect(page.get_by_text('不存在的演示问题？')).to_be_visible();page.get_by_role('button',name='驳回',exact=True).first.click();expect(page.get_by_text('当前状态下没有记录。')).to_be_visible();assert decisions[0]['approved_answer']=='核准答案' and decisions[1]['decision']=='驳回'
        page.goto(base_url.rstrip('/')+'/observability');expect(page.get_by_text('Langfuse 用量暂不可用')).to_be_visible();expect(page.locator('.run')).to_have_count(2);expect(page.locator('.run').first).to_contain_text('dataset d2');expect(page.locator('.run').nth(1)).to_contain_text('dataset d1');assert not errors;browser.close()
    return {'status':'passed','checks':['review list/detail','NULL/empty/Top3 snapshots','approval and rejection','usage unavailable','two-run trend']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-url',default='http://ch09.test');parser.add_argument('--output',type=Path,default=Path('data/ch09/reports/browser.json'));args=parser.parse_args();result=verify(args.base_url);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__': raise SystemExit(main())
