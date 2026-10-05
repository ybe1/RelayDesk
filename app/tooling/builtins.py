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
        """客户明确要求创建工单且问题描述齐全时调用，以生成待确认预览；调用本身不建单，只有客户确认预览后才实际写入。description必须来自客户原话，ticket_type为售后、投诉或咨询。"""
        from app.services.ticket_confirmation import execute_ticket_once
        return execute_ticket_once(context, {'description': description, 'ticket_type': ticket_type})
    return create_ticket
