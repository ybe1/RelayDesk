"""Restore complete history and close interrupted requests without execution."""
import json
from langchain_core.messages import ToolMessage


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
