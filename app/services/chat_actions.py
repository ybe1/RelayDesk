"""The original complaint button remains a confirmed, audited ticket action."""
import asyncio
import json
from threading import Lock, RLock
from weakref import WeakValueDictionary

from sqlalchemy import select
from app.db.models import ChatActionOffer
from app.repositories.conversations import get_or_create_conversation
from app.tools.business import TicketArgs
from app.tooling.builtins import bind_ticket
from app.tooling.contracts import CallContext, ToolDefinition
from app.tooling.permissions import issue_write_permit
from app.services.ticket_confirmation import TicketConfirmationService
from app.services.tool_executor import ToolExecutor

_LOCKS = WeakValueDictionary()
_LOCK_GUARD = Lock()


def _offer_lock(offer_id):
    with _LOCK_GUARD:
        lock = _LOCKS.get(offer_id)
        if lock is None:
            lock = RLock()
            _LOCKS[offer_id] = lock
        return lock


def confirm_ticket(session_factory, user_id, conversation_id, offer_id, description, ticket_type):
    params = TicketArgs(description=description, ticket_type=ticket_type).model_dump()
    if not description.strip() or len(description) > 4000:
        raise ValueError('invalid description')
    # Serializes same-process button retries; the durable claim also protects
    # independent processes and recovery after an uncertain write.
    with _offer_lock(offer_id):
        with session_factory() as session:
            get_or_create_conversation(session, user_id, conversation_id)
            offer = session.scalar(select(ChatActionOffer).where(
                ChatActionOffer.offer_id == offer_id).with_for_update())
            if offer is None:
                raise LookupError('offer not found')
            if offer.conversation_id != conversation_id:
                raise PermissionError('offer belongs to another conversation')
            if offer.action != 'ticket':
                raise ValueError('offer is not a ticket action')
            if offer.ticket_no is not None:
                if offer.confirmed_params != params:
                    raise ValueError('confirmed parameters cannot change')
                return TicketConfirmationService._result(offer)
            context = CallContext(conversation_id=conversation_id, user_id=user_id,
                turn_id=offer.turn_id, session_factory=session_factory, ticket_request=True)
            tool = bind_ticket(context)
            definition = ToolDefinition.from_tool(tool)
            call = {'name': 'create_ticket', 'id': 'button-'+offer_id, 'args': params}
            metadata = offer.defaults.get('_ch08')
            if metadata is not None:
                # Chat interrupt previews must resume their original workflow.
                if metadata.get('entry') != 'button' or metadata.get('call') != call:
                    raise ValueError('confirmation must resume its original workflow')
            else:
                metadata = {'entry': 'button', 'state': 'confirmed', 'call': call,
                            'definition_hash': definition.definition_hash}
                offer.defaults = {**offer.defaults, '_ch08': metadata}
            session.commit()
        context.confirmation = issue_write_permit(context, call, definition.definition_hash, offer_id)
        message = asyncio.run(ToolExecutor([tool], context=context).execute(call))
        payload = json.loads(message.content)
        if 'error' in payload:
            if payload['error']['code'] == 'permission_denied':
                raise PermissionError(payload['error']['message'])
            raise ValueError(payload['error']['message'])
        return payload['result']
