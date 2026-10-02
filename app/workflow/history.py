"""Restore complete history and close interrupted requests without execution."""
import json
from langchain_core.messages import ToolMessage

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
