import json
import logging
from dataclasses import asdict
from .tokens import estimate_messages

LOGGER=logging.getLogger('app.context')

def _dump(messages):
    return [{'id':m.id,'role':m.type,'content':m.content,
        'tool_calls':getattr(m,'tool_calls',None),'tool_call_id':getattr(m,'tool_call_id',None)} for m in messages]

def record_history(cid,turn_id,projection,budget):
    LOGGER.info('history_ctx %s',json.dumps({'conversation_id':cid,'turn_id':turn_id,
        'anchors':asdict(projection.anchors),'summary':projection.summary,
        'layer2':_dump(projection.layer2),'layer1':_dump(projection.layer1),
        'window_count':len(projection.layer2)+len(projection.layer1),
        'tokens_estimate':estimate_messages([*projection.layer2,*projection.layer1]),
        'omitted_ranges':projection.omitted_ranges,'budget':asdict(budget)},ensure_ascii=False))

def record_model(cid,turn_id,phase,messages,tools,budget):
    LOGGER.info('model_ctx %s',json.dumps({'conversation_id':cid,'turn_id':turn_id,'phase':phase,
        'messages':_dump(messages),'tools':tools,'window_count':len(messages),
        'tokens_estimate':estimate_messages(messages,tools),'budget':asdict(budget)},ensure_ascii=False,default=str))
