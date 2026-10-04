"""Durable confirmation and at-most-once dispatch for ticket writes."""
from uuid import uuid4

from sqlalchemy import select

from app.db.models import ChatActionOffer, Conversation
from app.repositories.chat_actions import issue_offer
from app.repositories.tickets import create_ticket
from app.tooling.permissions import WritePermit
from app.tools.business import TicketArgs


class TicketConfirmationService:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def _owned(self, session, user_id, conversation_id, offer_id):
        conversation = session.get(Conversation, conversation_id)
        if conversation is None or conversation.user_id != user_id:
            raise PermissionError("conversation ownership mismatch")
        offer = session.scalar(select(ChatActionOffer).where(
            ChatActionOffer.offer_id == offer_id).with_for_update())
        if offer is None:
            raise LookupError("confirmation not found")
        if offer.conversation_id != conversation_id or offer.action != 'ticket':
            raise PermissionError("confirmation ownership mismatch")
        return offer

    def prepare(self, context, call, definition_hash):
        if not context.ticket_request or call.get('name') != 'create_ticket':
            raise PermissionError("explicit ticket request required")
        params = TicketArgs(**call['args']).model_dump()
        if not params['description'].strip() or len(params['description']) > 4000:
            raise ValueError("invalid ticket description")
        metadata = {'call': call, 'definition_hash': definition_hash, 'state': 'prepared'}
        with self.session_factory() as session:
            conversation = session.get(Conversation, context.conversation_id)
            if conversation is None or conversation.user_id != context.user_id:
                raise PermissionError("conversation ownership mismatch")
            offer = issue_offer(session, context.conversation_id, context.turn_id,
                                'ticket', {**params, '_ch08': metadata})
            existing = offer.defaults.get('_ch08', {})
            if existing.get('call') != call or existing.get('definition_hash') != definition_hash:
                raise PermissionError("confirmation parameters cannot change")
            preview = {'kind': 'ticket_confirmation', 'confirmation_id': offer.offer_id,
                       'turn_id': context.turn_id, 'tool_call_id': call['id'], **params}
            session.commit()
            return preview

    def _transition(self, user_id, conversation_id, offer_id, target):
        with self.session_factory() as session:
            offer = self._owned(session, user_id, conversation_id, offer_id)
            metadata = dict(offer.defaults.get('_ch08', {}))
            state = metadata.get('state')
            if target == 'confirmed' and state in {'completed', 'executing'}:
                return
            if state == target:
                return
            if state not in {'prepared', 'confirmed'}:
                raise PermissionError("confirmation is no longer pending")
            metadata['state'] = target
            offer.defaults = {**offer.defaults, '_ch08': metadata}
            session.commit()

    def confirm(self, user_id, conversation_id, offer_id):
        self._transition(user_id, conversation_id, offer_id, 'confirmed')

    def cancel(self, user_id, conversation_id, offer_id):
        self._transition(user_id, conversation_id, offer_id, 'cancelled')

    def claim(self, context, call):
        permit = context.confirmation
        if not isinstance(permit, WritePermit):
            raise PermissionError("trusted confirmation required")
        with self.session_factory() as session:
            offer = self._owned(session, context.user_id, context.conversation_id, permit.offer_id)
            metadata = dict(offer.defaults.get('_ch08', {}))
            if (offer.turn_id != context.turn_id or metadata.get('call') != call or
                    not permit.allows(context, call, metadata.get('definition_hash'))):
                raise PermissionError("confirmation does not match call")
            state = metadata.get('state')
            if state == 'completed' and offer.ticket_no:
                return {'state': 'completed', 'result': self._result(offer)}
            if state == 'executing':
                return {'state': 'unknown'}
            if state != 'confirmed':
                raise PermissionError("ticket has not been confirmed")
            token = uuid4().hex
            metadata.update(state='executing', claim_token=token)
            offer.defaults = {**offer.defaults, '_ch08': metadata}
            session.commit()
            return {'state': 'claimed', 'token': token}

    @staticmethod
    def _result(offer):
        return {'ticket_no': offer.ticket_no, 'conversation_id': offer.conversation_id,
                'status': '待处理'}


def execute_ticket_once(context, args):
    permit = context.confirmation
    if not isinstance(permit, WritePermit):
        raise PermissionError("trusted confirmation required")
    call = {'name': 'create_ticket', 'id': permit.call_id, 'args': args}
    service = TicketConfirmationService(context.session_factory)
    claim = service.claim(context, call)
    if claim['state'] == 'completed':
        return claim['result']
    if claim['state'] == 'unknown':
        return {'error': {'code': 'outcome_unknown', 'message': '工单提交结果待核实，请勿重复提交。'}}
    # The durable claim precedes the write. A crash leaves an uncertain result,
    # never permission to dispatch again. Ticket and receipt commit together.
    with context.session_factory() as session:
        offer = service._owned(session, context.user_id, context.conversation_id, permit.offer_id)
        metadata = dict(offer.defaults['_ch08'])
        if metadata.get('claim_token') != claim['token'] or metadata['state'] != 'executing':
            raise PermissionError("ticket claim is no longer valid")
        ticket = create_ticket(session, context.conversation_id, **args)
        offer.ticket_no = ticket.ticket_no
        offer.confirmed_params = args
        metadata['state'] = 'completed'
        offer.defaults = {**offer.defaults, '_ch08': metadata}
        result = service._result(offer)
        session.commit()
        return result
