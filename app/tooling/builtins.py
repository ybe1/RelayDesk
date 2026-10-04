from app.tools.business import build_tools


def bind_builtins(context):
    """Production catalog. The historical logistics factory is never published."""
    return [tool for tool in build_tools(context.session_factory,context.conversation_id,
        raw_question=context.raw_question,budget=context.budget,model=context.model) if tool.name!='query_logistics']
