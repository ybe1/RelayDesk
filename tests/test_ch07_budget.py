import pytest

def test_demo_budget_is_derived():
    from app.context.contracts import ContextConfig
    from app.context.budget import calculate_budget
    cfg = ContextConfig(window=18000, output=2000, user_input=2000,
                        steps=3, tool_result=1200, rerank_k=5)
    b = calculate_budget(cfg)
    assert (b.fixed, b.transient, b.history, b.layer1, b.layer2) == (5550,6800,5650,3954,1695)
    assert calculate_budget(cfg, summary_tokens=800).history == 5450

def test_impossible_window_reports_breakdown():
    from app.context.contracts import ContextConfig
    from app.context.budget import validate_context_capacity
    with pytest.raises(ValueError, match='上下文预算不足'):
        validate_context_capacity(ContextConfig(window=1000), prefix_tokens=900)

def test_prefix_overflow_is_not_hidden():
    from app.context.contracts import ContextConfig
    from app.context.budget import validate_context_capacity
    with pytest.raises(ValueError, match='前缀'):
        validate_context_capacity(ContextConfig(), prefix_tokens=1601)

def test_invalid_config_is_rejected():
    from app.context.contracts import ContextConfig
    from app.context.budget import calculate_budget
    with pytest.raises(ValueError):
        calculate_budget(ContextConfig(steps=0))
