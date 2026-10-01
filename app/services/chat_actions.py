"""Only an explicit confirmed offer may invoke the existing ticket tool."""
import json
from sqlalchemy import select
from app.db.models import ChatActionOffer
from app.repositories.conversations import get_or_create_conversation
from app.tools.business import build_tools, TicketArgs


def confirm_ticket(session_factory, user_id, conversation_id, offer_id, description, ticket_type):
    params = TicketArgs(description=description, ticket_type=ticket_type).model_dump()
    if len(description) > 4000:
        raise ValueError("description too long")
    with session_factory() as session:
        get_or_create_conversation(session, user_id, conversation_id)
        offer = session.scalar(select(ChatActionOffer).where(ChatActionOffer.offer_id == offer_id).with_for_update())
        if offer is None:
            raise LookupError("offer not found")
        if offer.conversation_id != conversation_id:
            raise PermissionError("offer belongs to another conversation")
        if offer.action != "ticket":
            raise ValueError("offer is not a ticket action")
        if offer.ticket_no is not None:
            if offer.confirmed_params != params:
                raise ValueError("confirmed parameters cannot change")
            return {"ticket_no": offer.ticket_no, "conversation_id": conversation_id, "status": "待处理"}
        tool = next(t for t in build_tools(session_factory, conversation_id, ticket_session=session) if t.name == "create_ticket")
        result = json.loads(tool.invoke(params))
        offer.ticket_no = result["ticket_no"]
        offer.confirmed_params = params
        session.commit()
        return result
