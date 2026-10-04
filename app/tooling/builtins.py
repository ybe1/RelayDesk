from langchain_core.tools import tool
from app.tools.business import build_tools, TicketArgs


def bind_builtins(context):
    """Production catalog. The historical logistics factory is never published."""
    return [item for item in build_tools(context.session_factory,context.conversation_id,
        raw_question=context.raw_question,budget=context.budget,model=context.model)
        if item.name not in {'query_logistics','create_ticket'}] + [bind_ticket(context)]


def bind_ticket(context):
    @tool(args_schema=TicketArgs)
    def create_ticket(description: str, ticket_type: str):
        """客户明确要求创建人工工单且确认预览后提交；需真实的问题描述和工单类型。"""
        from app.services.ticket_confirmation import execute_ticket_once
        return execute_ticket_once(context, {'description': description, 'ticket_type': ticket_type})
    return create_ticket
