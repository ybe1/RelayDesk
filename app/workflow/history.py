"""Restore complete history and close interrupted requests without execution."""
import json
import re
from langchain_core.messages import ToolMessage


def is_history_recall(question):
    """Explicit questions about what was said, excluding current business actions."""
    if re.search(r'能(?:不能)?退|能否退|可以退|政策|规则|运费|邮费|保修|怎么(?:退|申请|维修)|申请(?:退款|退货|维修)|到哪|到账',question):
        return False
    prior=re.search(r'最开始|最早|刚才|之前|先前|前面|当时',question)
    facts=re.search(r'订单号|手机号|电话|诉求|记得|报过|留下|说过|提过|问过|聊过|怎么说|帮我做什么',question)
    return bool(prior and facts)


def understanding_payload(messages):
    """Only conversational fields enter understanding prompts, never UI metadata."""
    rows=[]
    for message in messages:
        row={'type':message.type,'content':message.content}
        if getattr(message,'tool_calls',None): row['tool_calls']=message.tool_calls
        if message.type=='tool': row['tool_call_id']=message.tool_call_id
        rows.append(row)
    return rows


def ordered_tool_history(messages):
    """Repair only the input projection; immutable checkpoint messages remain intact."""
    results={m.tool_call_id:m for m in messages if m.type=='tool'}
    ordered=[]
    for message in messages:
        if message.type=='tool': continue
        ordered.append(message)
        for call in getattr(message,'tool_calls',[]) or []:
            if call['id'] in results: ordered.append(results[call['id']])
    return ordered

def close_missing_results(messages):
    answered={m.tool_call_id for m in messages if m.type=='tool'}
    closed=[]
    for message in messages:
        for call in getattr(message,'tool_calls',[]) or []:
            if call['id'] in answered: continue
            closed.append(ToolMessage(content=json.dumps({'error':{'code':'interrupted',
                'message':'上次工具调用被中断，结果未知，未恢复执行'}},ensure_ascii=False),
                tool_call_id=call['id'],id=f"recovery:{message.id}:{call['id']}"))
            answered.add(call['id'])
    return closed

def merge_history(checkpoint,database):
    """Prefer checkpoint results; stable MySQL keys deduplicate stored messages."""
    known={m.id for m in checkpoint}
    messages=list(checkpoint)
    for m in database:
        if m.id not in known:
            messages.append(m)
            known.add(m.id)
    # First restoration from SQL is chronological. Existing checkpoint normally
    # already contains every SQL message; missing persisted writes are appended.
    return messages
