from dataclasses import asdict
from .contracts import ContextConfig, BudgetBreakdown


class ContextCapacityError(ValueError):
    """Safe, actionable input/context capacity message for the SSE boundary."""

def calculate_budget(config: ContextConfig, summary_tokens: int = 0) -> BudgetBreakdown:
    if any(isinstance(v,bool) or not isinstance(v,int) or v<=0 for v in asdict(config).values()):
        raise ValueError('上下文配置必须为正整数')
    margin = max(450, (config.window+49)//50)
    q = max(config.summary_reserve,summary_tokens)
    fixed = config.prefix+config.rerank_k*config.evidence_per_hit+q+config.output+margin
    transient = config.user_input+config.steps*(config.tool_result+config.call_overhead)
    steady = config.user_input+config.output+config.tool_result+config.call_overhead
    h = min(config.desired_turns*steady,config.window-fixed-transient)
    return BudgetBreakdown(fixed,transient,steady,h,max(0,7*h//10-1),max(0,3*h//10),q,margin)

def validate_context_capacity(config: ContextConfig, prefix_tokens: int) -> BudgetBreakdown:
    b = calculate_budget(config)
    if prefix_tokens > config.prefix:
        raise ValueError(f'上下文前缀超额: actual={prefix_tokens} allowance={config.prefix}')
    if b.history < b.steady:
        raise ValueError(f'上下文预算不足: {asdict(b)} window={config.window}')
    return b

def config_from_settings(settings) -> ContextConfig:
    return ContextConfig(window=settings.model_context_window, output=settings.max_output_tokens,
        user_input=settings.max_user_input_tokens,steps=settings.max_agent_steps,
        tool_result=settings.tool_result_max_tokens,rerank_k=settings.rerank_top_k,
        desired_turns=settings.context_desired_turns)

def cumulative_token_limit(config: ContextConfig, model_calls: int) -> int:
    """Independent turn cost guard: bounded calls, each bounded by its window."""
    return model_calls * (config.window - max(450,(config.window+49)//50))
