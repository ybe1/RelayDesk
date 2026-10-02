"""Headless UI verification with fully intercepted HTTP; no local listener."""
import json
from pathlib import Path
from urllib.parse import urlparse,parse_qs

def verify():
    from playwright.sync_api import sync_playwright,expect
    html=Path('app/static/index.html').read_text(encoding='utf-8')
    calls=[]; errors=[]; fail_history=False; fail_list=False
    def handle(route):
        nonlocal fail_history,fail_list
        request=route.request; parsed=urlparse(request.url); path=parsed.path
        calls.append((request.method,path))
        if path=='/': return route.fulfill(content_type='text/html',body=html)
        if path=='/api/conversations':
            if fail_list: return route.fulfill(status=500,body='offline')
            cursor=parse_qs(parsed.query).get('cursor')
            page={'items':[{'id':1,'first_question':'旧订单1001','has_summary':True}],'next_cursor':None} if cursor else {
                'items':[{'id':2,'first_question':'新商品问题','has_summary':False}],'next_cursor':'older'}
            return route.fulfill(json=page)
        if path.endswith('/messages'):
            if fail_history: return route.fulfill(status=500,body='offline')
            cid=int(path.split('/')[3])
            text='完整订单原话。'*40 if cid==1 else '新商品原话'
            return route.fulfill(json={'conversation_id':cid,'messages':[
                {'id':cid*10,'role':'user','content':text,'tool_calls':None,'citations':[],'actions':[]},
                {'id':cid*10+1,'role':'assistant','content':'完整答复 [1]','tool_calls':None,
                 'citations':[{'n':1,'chunk_id':5,'section_path':'演示/政策','answer_excerpt':'条款'}],
                 'actions':[{'action':'ticket','offer_id':'done','defaults':{},'confirmed':True,'ticket_no':'T-DEMO'}]}]})
        if path.endswith('/pending'): return route.fulfill(json={'pending':None})
        if path=='/api/kb/chunks/5': return route.fulfill(json={'section_path':'演示/政策','questions':'完整问题','answer':'完整来源原文'})
        return route.abort()
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        context=browser.new_context()
        context.route('**/*',handle)
        page=context.new_page(); page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto('http://ch07.test/')
        expect(page.locator('#conversation-sidebar')).to_be_visible()
        expect(page.locator('.conversation-item')).to_have_count(2)
        page.locator('[data-conversation-id="1"]').click()
        expect(page.locator('#messages')).to_contain_text('完整订单原话。'*40)
        expect(page.get_by_text('已摘要',exact=True)).to_be_visible()
        expect(page.get_by_role('button',name='工单已创建')).to_be_disabled()
        page.get_by_role('button',name='查看引用 1：演示/政策').click()
        expect(page.locator('#citation-answer')).to_have_text('完整来源原文')
        page.locator('#citation-close').click()
        page.get_by_role('button',name='回答有帮助').click()
        expect(page.get_by_text('已反馈',exact=True)).to_be_visible()
        page.locator('[data-conversation-id="2"]').click()
        expect(page.locator('#messages')).to_contain_text('新商品原话')
        page.locator('[data-conversation-id="1"]').click()
        expect(page.get_by_text('已反馈',exact=True)).to_be_visible()
        fail_history=True
        page.locator('[data-conversation-id="2"]').click()
        expect(page.locator('#messages')).to_contain_text('完整订单原话。')
        fail_history=False
        page.locator('#new-conversation').click()
        expect(page.locator('#messages')).not_to_contain_text('完整订单原话。')
        expect(page.locator('.conversation-item')).to_have_count(2)
        page.evaluate('setSending(true)')
        expect(page.locator('#new-conversation')).to_be_disabled()
        expect(page.locator('.conversation-item').first).to_be_disabled()
        page.evaluate('setSending(false)')
        fail_list=True;page.evaluate('refreshSidebar()')
        expect(page.locator('#send')).to_be_enabled()
        assert not any(method=='POST' for method,path in calls),calls
        assert not errors,errors
        browser.close()
    return {'status':'passed','scope':'mocked HTTP headless UI; no production server',
            'checks':['pagination','full_history','citation','feedback_restore','confirmed_action','failed_history','new_conversation','stream_lock','sidebar_failure'],
            'requests':len(calls)}

if __name__=='__main__': print(json.dumps(verify(),ensure_ascii=False))
