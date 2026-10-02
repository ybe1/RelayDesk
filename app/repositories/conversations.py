from sqlalchemy.orm import Session

from app.db.models import Conversation
import base64
import json
from datetime import datetime
from sqlalchemy import select,or_,and_
from app.db.models import Message,ChatActionOffer,RefundRequest

def _cursor(value):
    try:
        if len(value)>256: raise ValueError()
        timestamp,cid=json.loads(base64.urlsafe_b64decode(value).decode())
        if type(cid) is not int or cid<=0: raise ValueError()
        return datetime.fromisoformat(timestamp),cid
    except Exception as exc: raise ValueError('invalid conversation cursor') from exc

def list_owned(session,user_id,cursor=None,limit=50):
    first=select(Message.content).where(Message.conversation_id==Conversation.id,Message.role=='user').order_by(Message.id).limit(1).scalar_subquery()
    query=select(Conversation,first).where(Conversation.user_id==user_id)
    if cursor:
        stamp,cid=_cursor(cursor)
        query=query.where(or_(Conversation.created_at<stamp,and_(Conversation.created_at==stamp,Conversation.id<cid)))
    rows=list(session.execute(query.order_by(Conversation.created_at.desc(),Conversation.id.desc()).limit(limit+1)))
    items=[{'id':c.id,'created_at':c.created_at,'status':c.status,'first_question':(question or '新对话')[:120],
        'has_summary':bool(c.summary_upto_msg_id)} for c,question in rows[:limit]]
    next_cursor=None
    if len(rows)>limit:
        c=rows[limit-1][0]
        next_cursor=base64.urlsafe_b64encode(json.dumps([c.created_at.isoformat(),c.id]).encode()).decode()
    return {'items':items,'next_cursor':next_cursor}

def owned_history(session,user_id,cid):
    get_or_create_conversation(session,user_id,cid)  # Existing id: check only, never create.
    rows=list(session.scalars(select(Message).where(Message.conversation_id==cid).order_by(Message.id)))
    offers=list(session.scalars(select(ChatActionOffer).where(ChatActionOffer.conversation_id==cid)))
    refunds={r.offer_id:r for r in session.scalars(select(RefundRequest).where(RefundRequest.conversation_id==cid))}
    result=[]
    for row in rows:
        actions=[]
        if row.role=='assistant' and row.message_key and row.message_key.endswith(':final'):
            turn=row.message_key[:-len(':final')]
            for offer in offers:
                if offer.turn_id!=turn: continue
                item={'action':offer.action,'offer_id':offer.offer_id,'defaults':offer.defaults,
                    'ticket_no':offer.ticket_no,'confirmed':bool(offer.confirmed_params)}
                if offer.offer_id in refunds:
                    item['application_no']=refunds[offer.offer_id].application_no
                    item['confirmed']=True
                actions.append(item)
        result.append({'id':row.id,'role':row.role,'content':row.content,'tool_calls':row.tool_calls,
            'created_at':row.created_at,'citations':[],'actions':actions,'message_key':row.message_key})
    return {'conversation_id':cid,'messages':result}


def get_or_create_conversation(session: Session, user_id: str, conversation_id: int | None = None) -> Conversation:
    if conversation_id is None:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        session.flush()
        return conversation
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise LookupError("conversation not found")
    if conversation.user_id != user_id:
        raise PermissionError("conversation belongs to another user")
    return conversation
