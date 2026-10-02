"""Current-session provenance and stable demonstration order cards."""
import json
import re
from app.services.chat import trim_turns
from .routing_contracts import Resolution

ORDER_PATTERNS = (
    r'(?:订单|单号)\s*[:：#]?\s*([A-Za-z0-9_-]{1,64})',
    r'(?<![A-Za-z0-9])([0-9]{3,20})\s*(?:那单|这单|的订单)',
)


def _explicit_orders(text):
    if not isinstance(text,str): return set()
    return {match for pattern in ORDER_PATTERNS for match in re.findall(pattern,text)}


def recent_turns(messages,max_turns=6,token_budget=2000):
    starts = [i for i,m in enumerate(messages) if m.type=='human']
    window = messages[starts[-max_turns]:] if len(starts)>=max_turns else list(messages)
    return trim_turns(window,token_budget)


def trusted_order_ids(raw,history):
    past = set()
    calls = {}
    for message in history:
        if message.type=='human':
            past.update(_explicit_orders(message.content))
        elif message.type=='ai':
            calls.update({call['id']:call for call in getattr(message,'tool_calls',[])})
        elif message.type=='tool':
            call = calls.get(message.tool_call_id)
            if not call or call['name'] not in {'query_order','query_logistics'}:
                continue
            try:
                data = json.loads(message.content)
                if 'result' in data:
                    data = json.loads(data['result']) if isinstance(data['result'],str) else data['result']
                order_id = data.get('order_id')
                if isinstance(order_id,str) and order_id == call['args'].get('order_id') and re.fullmatch(r'[A-Za-z0-9_-]{1,64}',order_id):
                    past.add(order_id)
            except (ValueError,TypeError,AttributeError):
                pass
    return {'current':_explicit_orders(raw),'history':past}


def validate_resolution(result,raw,history):
    sources = trusted_order_ids(raw,history)
    if result.order_id is None:
        return result
    valid = result.order_id in sources.get(result.order_source,set())
    # A clear current reference wins; an ambiguous historical set must not be
    # resolved by arbitrarily taking the most recently mentioned number.
    if result.order_source=='history' and (len(sources['history'])!=1 or sources['current']):
        valid = False
    if result.order_source=='current' and len(sources['current'])!=1:
        valid = False
    if valid:
        return result
    return Resolution(question=raw,reference_status='ambiguous',scope=result.scope,
                      order_id=None,order_source='none')


def demo_order(order_id):
    from app.tools.business import _mock_order_snapshot
    snapshot = _mock_order_snapshot(order_id)
    return {'mock':True,'order_id':order_id,**{k:v for k,v in snapshot.items() if k not in {'logistics_status','location'}}}


def demo_order_options():
    return [demo_order(order_id) for order_id in ('1001','1002','1003')]
