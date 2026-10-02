"""One bounded model request, followed by server-side slot provenance checks."""
import json
from langchain_core.messages import SystemMessage, HumanMessage
from app.core.workflow_prompts import RESOLUTION_PROMPT
from .model import BudgetedModel
from .routing_contracts import Resolution
from .order_context import recent_turns, validate_resolution
from app.context.tokens import estimate_messages
from .history import understanding_payload


async def resolve_question(model,raw_question,history,budget,shared_history=False):
    history=list(history) if shared_history else recent_turns(history)
    # History is nested inside one HumanMessage here, so the generic model
    # wrapper cannot see its turn boundaries. Fit complete historical turns
    # before nesting them, using the same conservative request estimate.
    while True:
        payload={'history':understanding_payload(history),'current_question':raw_question}
        messages=[SystemMessage(content=RESOLUTION_PROMPT),HumanMessage(content=json.dumps(payload,ensure_ascii=False))]
        inputs=estimate_messages(messages)
        if shared_history or not history or (estimate_messages(history)<=2000 and budget.tokens+inputs+512<=budget.limits.tokens):
            break
        next_turn=next((i for i,message in enumerate(history) if i>0 and message.type=='human'),len(history))
        history=history[next_turn:]
    response=await BudgetedModel(model,budget,512).ainvoke(
        messages,
        response_format={'type':'json_object'},extra_body={'thinking':{'type':'disabled'}})
    try:
        result=Resolution.model_validate(json.loads(response.content))
    except (ValueError,TypeError):
        return Resolution(question=raw_question,reference_status='ambiguous',scope='order_specific',order_source='none')
    return validate_resolution(result,raw_question,history)
