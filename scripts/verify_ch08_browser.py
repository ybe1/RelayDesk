import json
from pathlib import Path
from urllib.parse import urlparse


def verify():
    from playwright.sync_api import sync_playwright,expect
    html=Path('app/static/index.html').read_text(encoding='utf-8')
    report_dir=Path('data/ch08/reports');report_dir.mkdir(parents=True,exist_ok=True)
    calls=[]; counts={'tickets':0}; pending={}
    def sse(events):
        return ''.join((f'data: {json.dumps(data,ensure_ascii=False)}\n\n' if kind=='delta'
            else f'event: {kind}\ndata: {json.dumps(data,ensure_ascii=False)}\n\n' if kind!='done'
            else 'data: [DONE]\n\n') for kind,data in events)
    def handle(route):
        req=route.request;path=urlparse(req.url).path;calls.append((req.method,path))
        if path=='/': return route.fulfill(content_type='text/html',body=html)
        if path=='/api/chat' and req.method=='POST':
            body=req.post_data_json;cid=body.get('conversation_id') or 81
            if body.get('message')=='继续普通聊天':
                pending.pop(cid,None)
                return route.fulfill(content_type='text/event-stream',body=sse([
                    ('conversation',{'conversation_id':cid}),('delta',{'delta':'可以继续普通聊天。'}),('done',None)]))
            if body.get('message')=='投诉':
                return route.fulfill(content_type='text/event-stream',body=sse([
                    ('conversation',{'conversation_id':cid}),('actions',{'conversation_id':cid,'actions':[
                        {'action':'handoff','offer_id':'handoff-'+str(cid)},
                        {'action':'ticket','offer_id':'legacy-ticket-'+str(cid),
                         'defaults':{'description':'用户明确提交后才建单','ticket_type':'投诉'}}]}),('done',None)]))
            if body.get('message')=='我要查询订单':
                return route.fulfill(content_type='text/event-stream',body=sse([
                    ('conversation',{'conversation_id':cid}),('order_selection',{'conversation_id':cid,
                        'selection_id':'selection-'+str(cid),'turn_id':'turn-order',
                        'options':[{'order_id':'1001','product_name':'咖啡机','status':'待发货','amount':'299'}]}),
                    ('done',None)]))
            card={'kind':'ticket_confirmation','conversation_id':cid,'confirmation_id':'confirm-'+str(cid),
                'turn_id':'turn-81','tool_call_id':'call-81','ticket_type':'售后',
                'description':'订单81包裹破损 <img src=x onerror=alert(1)>'}
            pending[cid]=card
            return route.fulfill(content_type='text/event-stream',body=sse([
                ('conversation',{'conversation_id':cid}),('ticket_confirmation',card),('done',None)]))
        if path=='/api/chat/ticket-confirmation' and req.method=='POST':
            body=req.post_data_json;card=pending.get(body['conversation_id'])
            assert body['confirmation_id']==card['confirmation_id'] and 'description' not in body
            if body['decision']=='confirm': counts['tickets']+=1;text='已为您创建工单 T20261005001。'
            else:text='已取消工单提交，没有创建工单。'
            pending.pop(body['conversation_id'],None)
            return route.fulfill(content_type='text/event-stream',body=sse([
                ('conversation',{'conversation_id':body['conversation_id']}),('delta',{'delta':text}),('done',None)]))
        if path=='/api/chat/selection' and req.method=='POST':
            assert req.post_data_json['order_id']=='1001'
            return route.fulfill(content_type='text/event-stream',body=sse([
                ('delta',{'delta':'已选择订单1001。'}),('done',None)]))
        if path.endswith('/pending'):
            cid=int(path.split('/')[-2]);return route.fulfill(json={'pending':pending.get(cid)})
        if path=='/api/conversations': return route.fulfill(json={'items':[],'next_cursor':None})
        if path.endswith('/messages'):
            cid=int(path.split('/')[-2])
            return route.fulfill(json={'conversation_id':cid,'messages':[]})
        return route.fulfill(status=404,body='not found')
    with sync_playwright() as p:
        browser=p.chromium.launch(channel='msedge',headless=True)
        context=browser.new_context();context.route('**/*',handle)
        page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto('http://ch08.test/');page.locator('#input').fill('包裹破损，帮我建个工单');page.locator('#send').click()
        card=page.locator('.ticket-preview');expect(card).to_be_visible()
        expect(card).to_contain_text('工单类型：售后');expect(card).to_contain_text('<img src=x onerror=alert(1)>')
        assert card.locator('img').count()==0
        page.screenshot(path=str(report_dir/'ticket-preview.png'),full_page=True)
        confirm_button=page.get_by_role('button',name='确认提交')
        confirm_button.click()
        expect(page.get_by_text('T20261005001')).to_be_visible();assert counts['tickets']==1
        confirm_button.evaluate("el => el.dispatchEvent(new MouseEvent('click', {bubbles: true}))")
        assert counts['tickets']==1
        page.screenshot(path=str(report_dir/'ticket-confirmed.png'),full_page=True)
        page.locator('#new-conversation').click()
        page.locator('#input').fill('订单82包裹丢失，帮我建个工单');page.locator('#send').click()
        expect(page.locator('.ticket-preview')).to_be_visible()
        page.get_by_role('button',name='取消',exact=True).click()
        expect(page.get_by_text('没有创建工单')).to_be_visible();assert counts['tickets']==1
        page.screenshot(path=str(report_dir/'ticket-cancelled.png'),full_page=True)
        page.locator('#new-conversation').click()
        page.locator('#input').fill('订单83收到破损件，帮我建个工单');page.locator('#send').click()
        expect(page.locator('.ticket-preview')).to_be_visible()
        page.reload();expect(page.locator('.ticket-preview')).to_be_visible()
        expect(page.get_by_role('button',name='确认提交')).to_be_enabled()
        stale_confirm=page.get_by_role('button',name='确认提交')
        page.locator('#input').fill('继续普通聊天');page.locator('#send').click()
        expect(stale_confirm).to_be_disabled()
        expect(page.get_by_text('可以继续普通聊天。')).to_be_visible()
        page.locator('#new-conversation').click()
        page.locator('#input').fill('投诉');page.locator('#send').click()
        handoff=page.get_by_role('button',name='转人工');expect(handoff).to_be_visible()
        expect(page.get_by_role('button',name='建工单')).to_be_visible()
        handoff.click();expect(page.get_by_role('button',name='已转接人工客服')).to_be_disabled()
        expect(page.get_by_text('您好，我是客服小猫，请问有什么可以帮您的')).to_be_visible()
        assert counts['tickets']==1
        page.locator('#new-conversation').click()
        page.locator('#input').fill('我要查询订单');page.locator('#send').click()
        order=page.get_by_role('button',name='订单 1001 · 咖啡机');expect(order).to_be_visible()
        order.click();expect(page.get_by_text('已选择订单1001。')).to_be_visible()
        assert counts['tickets']==1 and not errors
        browser.close()
    return {'status':'passed','scope':'headless Edge, all HTTP intercepted, no local server/browser session',
        'checks':['preview and safe text rendering','confirm submits once with ticket number','cancel creates no ticket',
                  'pending preview restores after reload','new message invalidates stale ticket preview',
                  'legacy complaint handoff/ticket buttons remain independent','order card remains functional'],'requests':len(calls),
        'screenshots':[str(report_dir/name) for name in ('ticket-preview.png','ticket-confirmed.png','ticket-cancelled.png')]}


if __name__=='__main__':
    result=verify()
    Path('data/ch08/reports/browser.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
