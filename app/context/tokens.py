"""Conservative shared estimate: CJK one per character, ASCII four per token.

Estimates are not billing usage; actual provider usage is logged separately.
"""
import json
import math

def estimate_text(text: str) -> int:
    text = str(text)
    non_ascii = sum(ord(c) > 127 for c in text)
    return non_ascii + math.ceil((len(text)-non_ascii)/4)

def estimate_messages(messages, tools=()) -> int:
    total = 0
    for message in messages:
        total += 8 + estimate_text(message.content)
        if getattr(message, 'tool_calls', None):
            total += estimate_text(json.dumps(message.tool_calls, ensure_ascii=False))
        if getattr(message, 'tool_call_id', None):
            total += estimate_text(message.tool_call_id)
    if tools:
        total += estimate_text(json.dumps(list(tools),ensure_ascii=False,default=str))
    return total
