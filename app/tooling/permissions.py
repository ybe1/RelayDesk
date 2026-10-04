"""Process-local capability; never accepted from JSON, model output or a browser."""
import hashlib
import json
from dataclasses import dataclass

_AUTHORITY=object()


def argument_hash(args):
    return hashlib.sha256(json.dumps(args,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class WritePermit:
    authority:object
    user_id:str
    conversation_id:int
    call_id:str
    args_hash:str
    definition_hash:str
    offer_id:str

    def allows(self,context,call,definition_hash):
        return (self.authority is _AUTHORITY and context.ticket_request and
            (self.user_id,self.conversation_id,self.call_id,self.args_hash,self.definition_hash)==
            (context.user_id,context.conversation_id,call.get('id'),argument_hash(call.get('args',{})),definition_hash))


def issue_write_permit(context,call,definition_hash,offer_id):
    return WritePermit(_AUTHORITY,context.user_id,context.conversation_id,call['id'],
                       argument_hash(call['args']),definition_hash,offer_id)
