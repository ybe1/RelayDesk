from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import pytest
from sqlalchemy import delete,select
from app.db.session import SessionLocal
from app.db.models import Conversation,ChatActionOffer


@pytest.fixture
def refund_case():
    from app.db.models import RefundRequest
    from app.repositories.chat_actions import issue_offer
    from app.repositories.conversations import get_or_create_conversation
    from app.workflow.order_context import demo_order
    user = 'ch06-refund-'+uuid4().hex[:12]
    with SessionLocal() as session:
        cid = get_or_create_conversation(session,user).id
        defaults = {'order_id':'1001','order_snapshot':demo_order('1001'),
            'refund_assessment':{'status':'eligible','reason':'演示政策支持','evidence_numbers':[1]},
            'evidence':[{'n':1,'chunk_id':1,'answer':'七日内可申请退货'}]}
        offer = issue_offer(session,cid,'refund-turn','refund',defaults)
        offer_id = offer.offer_id
        session.commit()
    yield user,cid,offer_id
    with SessionLocal() as session:
        session.execute(delete(RefundRequest).where(RefundRequest.conversation_id==cid))
        session.execute(delete(ChatActionOffer).where(ChatActionOffer.conversation_id==cid))
        session.execute(delete(Conversation).where(Conversation.id==cid))
        session.commit()


def test_duplicate_submit_and_changed_reason(refund_case):
    from app.services.refunds import confirm_refund
    from app.db.models import RefundRequest
    user,cid,offer = refund_case
    first = confirm_refund(SessionLocal,user,cid,offer,'不喜欢/不需要')
    assert confirm_refund(SessionLocal,user,cid,offer,'不喜欢/不需要') == first
    with pytest.raises(ValueError): confirm_refund(SessionLocal,user,cid,offer,'商品质量问题')
    with SessionLocal() as session:
        rows = list(session.scalars(select(RefundRequest).where(RefundRequest.conversation_id==cid)))
        assert len(rows)==1 and rows[0].order_id=='1001' and rows[0].is_mock
        assert session.get(Conversation,cid).status=='进行中'


def test_concurrent_submit_has_one_application(refund_case):
    from app.services.refunds import confirm_refund
    user,cid,offer = refund_case
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(confirm_refund,SessionLocal,user,cid,offer,'其他') for _ in range(2)]
        results = [call.result(timeout=10) for call in calls]
    assert results[0]['application_no']==results[1]['application_no']


def test_wrong_owner_and_invalid_reason_write_nothing(refund_case):
    from app.services.refunds import confirm_refund
    from app.db.models import RefundRequest
    user,cid,offer = refund_case
    with pytest.raises(PermissionError): confirm_refund(SessionLocal,'other',cid,offer,'其他')
    with pytest.raises(ValueError): confirm_refund(SessionLocal,user,cid,offer,'任意原因')
    with SessionLocal() as session:
        assert not list(session.scalars(select(RefundRequest).where(RefundRequest.conversation_id==cid)))


@pytest.mark.parametrize('mutation',['non_refund','no_evidence','needs_review','wrong_order'])
def test_invalid_server_offer_cannot_be_submitted(refund_case,mutation):
    from app.services.refunds import confirm_refund
    user,cid,offer_id = refund_case
    with SessionLocal() as session:
        offer = session.get(ChatActionOffer,offer_id)
        defaults = dict(offer.defaults)
        if mutation=='non_refund': offer.action='handoff'
        elif mutation=='no_evidence': defaults['evidence']=[]
        elif mutation=='needs_review': defaults['refund_assessment']={'status':'needs_review','reason':'未知','evidence_numbers':[1]}
        else: defaults['order_id']='1002'
        offer.defaults = defaults
        session.commit()
    with pytest.raises(ValueError): confirm_refund(SessionLocal,user,cid,offer_id,'其他')


def test_request_cannot_override_order_or_amount():
    from app.api.refunds import RefundConfirmation
    with pytest.raises(ValueError): RefundConfirmation(user_id='u',conversation_id=1,offer_id='o',reason='其他',order_id='1002')
    with pytest.raises(ValueError): RefundConfirmation(user_id='u',conversation_id=1,offer_id='o',reason='其他',amount=999)


def test_commit_failure_rolls_back_application_and_confirmation(refund_case,monkeypatch):
    from app.services.refunds import confirm_refund
    from app.db.models import RefundRequest
    from sqlalchemy.orm import Session
    user,cid,offer_id = refund_case
    with monkeypatch.context() as patch:
        def fail_commit(self): raise RuntimeError('injected commit failure')
        patch.setattr(Session,'commit',fail_commit)
        with pytest.raises(RuntimeError,match='injected'):
            confirm_refund(SessionLocal,user,cid,offer_id,'其他')
    with SessionLocal() as session:
        assert not list(session.scalars(select(RefundRequest).where(RefundRequest.conversation_id==cid)))
        assert session.get(ChatActionOffer,offer_id).confirmed_params is None


def test_malformed_evidence_is_rejected_as_invalid_snapshot():
    from app.services.refunds import validate_refund_defaults
    with pytest.raises(ValueError):
        validate_refund_defaults({'refund_assessment':{'status':'eligible','reason':'test','evidence_numbers':[1]},'evidence':None})

