from contextlib import asynccontextmanager
from app.workflow.runtime import checkpoint_runtime
from app.workflow.graph import build_workflow
from app.workflow.service import WorkflowChatService
from app.services.chat import DatabaseChatStore
from app.db.session import SessionLocal
from tests.ch06_fakes import JSONSequenceModel
from tests.test_ch05_gate import result


@asynccontextmanager
async def service_case(tmp_path,resuming=False):
    before=[{'question':'我要退款','reference_status':'clear','scope':'order_specific','order_id':None,'order_source':'none'},
        {'intent':'退款退货','confidence':0.95}]
    after=[{'queries':['我要退款','订单退货政策']},
        {'kind':'answer','tool_calls':[],'actions':[],'reason':'证据够',
         'refund_assessment':{'status':'needs_review','reason':'未知商品状态','evidence_numbers':[1]}}]
    async def retrieve(question,budget): return result()
    async def retrieve_queries(question,queries,budget): return result()
    async with checkpoint_runtime(tmp_path/'resume.sqlite') as saver:
        yield WorkflowChatService(build_workflow(checkpointer=saver),DatabaseChatStore(SessionLocal),
            model_factory=lambda:JSONSequenceModel(after if resuming else before+after),
            retriever=retrieve,policy_retriever=retrieve_queries)


def remove_case(cid):
    from sqlalchemy import delete
    from app.db.models import RefundRequest,ChatActionOffer,Message,LowConfidenceQuestion,Ticket,Conversation
    with SessionLocal() as session:
        for cls in [RefundRequest,ChatActionOffer,Message,LowConfidenceQuestion,Ticket]:
            session.execute(delete(cls).where(cls.conversation_id==cid))
        session.execute(delete(Conversation).where(Conversation.id==cid)); session.commit()
