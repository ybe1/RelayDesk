from sqlalchemy import select
from app.db.models import Conversation,ConversationSummary
from app.context.contracts import LayerAnchors,SummarySegment

class SummaryRepository:
    def __init__(self,session_factory): self.session_factory=session_factory

    @staticmethod
    def _segment(row):
        return SummarySegment(row.seq,row.from_msg_id,row.upto_msg_id,row.content)

    def load(self,cid):
        with self.session_factory() as session:
            conv=session.get(Conversation,cid)
            if conv is None: raise ValueError('conversation not found')
            segments=session.scalars(select(ConversationSummary).where(
                ConversationSummary.conversation_id==cid).order_by(ConversationSummary.seq))
            return LayerAnchors(conv.summary_upto_msg_id or 0,conv.layer1_from_msg_id or 0),[self._segment(s) for s in segments]

    def advance_layer1(self,cid,expected,target):
        with self.session_factory() as session,session.begin():
            conv=session.scalar(select(Conversation).where(Conversation.id==cid).with_for_update())
            if conv is None or (conv.layer1_from_msg_id or 0)!=expected or target<expected:
                return False
            conv.layer1_from_msg_id=target
            return True

    def append(self,cid,expected_summary_upto,upto,content,from_msg_id=None):
        with self.session_factory() as session,session.begin():
            conv=session.scalar(select(Conversation).where(Conversation.id==cid).with_for_update())
            if conv is None or (conv.summary_upto_msg_id or 0)!=expected_summary_upto:
                return None
            if upto<=expected_summary_upto or upto>(conv.layer1_from_msg_id or 0): return None
            first=expected_summary_upto+1 if from_msg_id is None else from_msg_id
            if not expected_summary_upto<first<=upto: return None
            rows=list(session.scalars(select(ConversationSummary).where(
                ConversationSummary.conversation_id==cid).order_by(ConversationSummary.seq)))
            segment=None
            if content:
                row=ConversationSummary(conversation_id=cid,seq=max((r.seq for r in rows),default=0)+1,
                    from_msg_id=first,upto_msg_id=upto,content=content)
                session.add(row)
                rows.append(row)
                segment=self._segment(row)
            conv.summary='\n'.join(r.content for r in rows) or None
            conv.summary_upto_msg_id=upto
            session.flush()
            return segment
