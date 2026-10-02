"""Complete-turn, ID-anchored projections. Source messages stay immutable."""
import json
from langchain_core.messages import trim_messages
from .tokens import estimate_messages,estimate_text
from .contracts import HistoryRecord,HistoryProjection,LayerAnchors,SummaryBatch

def group_turns(records):
    turns=[]
    for record in records:
        if record.message.type=='human' or not turns: turns.append([])
        turns[-1].append(record)
    return [tuple(t) for t in turns]

def _payload(content):
    try:
        value=json.loads(content)
        if isinstance(value,dict) and 'result' in value:
            nested=value['result']
            return json.loads(nested) if isinstance(nested,str) else nested
        return value
    except (ValueError,TypeError): return {}

def tool_marker(message,name='tool'):
    data=_payload(message.content)
    base=f"工具{name} call={message.tool_call_id} "
    if estimate_text(base)>80:
        return '工具结果（标识过长，详见checkpoint原始记录）'
    facts={};omitted=False
    for key in ('order_id','product_id','status','order_status','logistics_status','mock'):
        if not isinstance(data,dict) or key not in data: continue
        candidate={**facts,key:data[key]}
        if estimate_text(base+json.dumps(candidate,ensure_ascii=False))<=100: facts=candidate
        else: omitted=True
    return base+json.dumps(facts,ensure_ascii=False)+('（部分字段过长，详见原始记录）' if omitted else '')

def semi_compress(turn):
    names={call['id']:call['name'] for r in turn for call in getattr(r.message,'tool_calls',[]) or []}
    result=[]
    for record in turn:
        message=record.message
        if message.type=='human': content=message.content
        elif message.type=='tool': content=tool_marker(message,names.get(message.tool_call_id,'tool'))
        else:
            content=str(message.content)
            if len(content)>80: content=content[:80]+'（已截短）'
        result.append(message.model_copy(update={'content':content}))
    return tuple(result)

def _flatten(turns): return [r.message for t in turns for r in t]

def _trim_turns(turns,max_tokens):
    messages=[m for turn in turns for m in semi_compress(turn)]
    trimmed=trim_messages(messages,max_tokens=max_tokens,token_counter=estimate_messages,
        strategy='last',start_on='human',allow_partial=False)
    candidate_ids={m.id for m in trimmed}
    # trim_messages handles messages, not entire turns. Retain a turn only when
    # every one of its messages survived, including all tool observations.
    kept=[]; omitted=[]
    for turn in turns:
        if all(r.message.id in candidate_ids for r in turn): kept.extend(semi_compress(turn))
        else: omitted.append((min(r.owner_msg_id for r in turn),max(r.owner_msg_id for r in turn)))
    return tuple(kept),tuple(omitted)

def project_history(records,anchors,budget,summary):
    if anchors.summary_upto<0 or anchors.layer1_from<anchors.summary_upto:
        raise ValueError('invalid context anchors')
    turns=group_turns(records)
    recent=[t for t in turns if max(r.owner_msg_id for r in t)>anchors.layer1_from]
    boundary=anchors.layer1_from
    while recent and estimate_messages(_flatten(recent))>budget.layer1:
        old=recent.pop(0)
        boundary=max(boundary,max(r.owner_msg_id for r in old))
    middle=[t for t in turns if anchors.summary_upto<max(r.owner_msg_id for r in t)<=boundary]
    all_middle=tuple(m for t in middle for m in semi_compress(t))
    tokens=estimate_messages(all_middle)
    layer2,omitted=_trim_turns(middle,budget.layer2) if tokens>budget.layer2 else (all_middle,())
    return HistoryProjection(LayerAnchors(anchors.summary_upto,boundary),tuple(_flatten(recent)),
        layer2,summary,tokens,omitted)

def prepare_summary_batch(cid,records,anchors,segments):
    source=tuple(r for r in records if anchors.summary_upto<r.owner_msg_id<=anchors.layer1_from)
    return SummaryBatch(cid,anchors.summary_upto,anchors.layer1_from,source,tuple(segments))

def history_records(messages,message_db_ids):
    from app.workflow.history import ordered_tool_history
    messages=ordered_tool_history(messages)
    call_owners={call['id']:message_db_ids.get(m.id,0) for m in messages for call in getattr(m,'tool_calls',[]) or []}
    records=[]; turn=''
    for message in messages:
        if message.type=='human': turn=message.id or str(len(records))
        owner=call_owners.get(message.tool_call_id,0) if message.type=='tool' else message_db_ids.get(message.id,0)
        if owner: records.append(HistoryRecord(message,owner,turn))
    return records

def cap_tool_result(message,max_tokens):
    if message.type!='tool' or estimate_messages([message])<=max_tokens: return message
    data=_payload(message.content)
    # Order snapshots are compact already; only bulky fields are projected.
    # If critical structured data itself does not fit, report unavailable rather
    # than drop qualification conditions and invite an eligibility guess.
    if isinstance(data,dict) and 'evidence' in data:
        data={k:v for k,v in data.items() if k not in {'evidence','citations'}}
        data['projection_notice']='证据正文请使用本轮资料；工具原文已保存在checkpoint。'
    content=json.dumps({'result':data},ensure_ascii=False)
    if estimate_text(content)+16>max_tokens:
        content=json.dumps({'error':{'code':'context_projection_limit','message':'工具数据超出输入额度，请人工核实，不能据此确认资格。'}},ensure_ascii=False)
    return message.model_copy(update={'content':content})
