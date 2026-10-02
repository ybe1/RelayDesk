"""User-confirmed demonstration applications; no payment operations."""
import json
from datetime import datetime
from uuid import uuid4
from sqlalchemy import select
from app.db.models import ChatActionOffer,RefundRequest
from app.repositories.conversations import get_or_create_conversation
from app.workflow.routing_contracts import RefundAssessment

REFUND_REASONS = ('不喜欢/不需要','商品质量问题','商品与描述不符','错发/漏发','其他')


def validate_refund_defaults(defaults):
    if not isinstance(defaults,dict): raise ValueError('invalid refund snapshot')
    assessment = RefundAssessment.model_validate(defaults.get('refund_assessment'))
    order = defaults.get('order_snapshot',{})
    evidence = defaults.get('evidence',[])
    if not isinstance(evidence,list): raise ValueError('invalid refund evidence')
    numbers = {item.get('n') for item in evidence if isinstance(item,dict) and item.get('answer') and item.get('chunk_id')}
    if (assessment.status!='eligible' or not numbers or not set(assessment.evidence_numbers).issubset(numbers)
        or not isinstance(order,dict) or order.get('mock') is not True
        or not isinstance(defaults.get('order_id'),str) or not defaults['order_id']
        or order.get('order_id')!=defaults['order_id']):
        raise ValueError('refund suggestion is not eligible')
    return json.loads(json.dumps(defaults,ensure_ascii=False))


def _result(row):
    return {'application_no':row.application_no,'conversation_id':row.conversation_id,'order_id':row.order_id,
            'status':row.status,'is_mock':True,'message':'演示申请已提交'}


def confirm_refund(session_factory,user_id,conversation_id,offer_id,reason):
    if reason not in REFUND_REASONS: raise ValueError('invalid refund reason')
    with session_factory() as session:
        get_or_create_conversation(session,user_id,conversation_id)
        offer = session.scalar(select(ChatActionOffer).where(ChatActionOffer.offer_id==offer_id).with_for_update())
        if offer is None: raise LookupError('offer not found')
        if offer.conversation_id!=conversation_id: raise PermissionError('offer belongs to another conversation')
        if offer.action!='refund': raise ValueError('offer is not a refund action')
        # Current locking read is essential after a concurrent commit under
        # MySQL REPEATABLE READ; an ordinary read could see an old snapshot.
        existing = session.scalar(select(RefundRequest).where(RefundRequest.offer_id==offer_id).with_for_update())
        if existing is not None:
            if existing.reason!=reason: raise ValueError('confirmed reason cannot change')
            return _result(existing)
        snapshot = validate_refund_defaults(offer.defaults)
        row = RefundRequest(application_no='R'+datetime.now().strftime('%Y%m%d')+uuid4().hex[:16],
            offer_id=offer_id,conversation_id=conversation_id,order_id=snapshot['order_id'],reason=reason,
            status='submitted',is_mock=True,snapshot=snapshot)
        session.add(row)
        offer.confirmed_params={'reason':reason}
        session.flush()
        result = _result(row)
        session.commit()
        return result
