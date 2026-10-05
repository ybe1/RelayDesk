import pytest
from langchain_core.messages import SystemMessage
from app.context.contracts import ContextConfig
from app.context.budget import calculate_budget,ContextCapacityError


def test_dynamic_schemas_charge_history_and_summary_independently():
    from app.context.budget import with_tool_prefix
    cfg=ContextConfig(window=40000,output=2000,user_input=2000,steps=3,tool_result=1200,rerank_k=5)
    schema={'type':'function','function':{'name':'large_tool','description':'工具说明'*3000,'parameters':{'type':'object'}}}
    adjusted=with_tool_prefix(cfg,[SystemMessage(content='客服')],[schema])
    assert adjusted.prefix>cfg.prefix
    assert calculate_budget(cfg,900).history-calculate_budget(adjusted,900).history==adjusted.prefix-cfg.prefix
    assert calculate_budget(adjusted,900).summary==900
    assert with_tool_prefix(cfg,[SystemMessage(content='客服')],[schema])==adjusted


def test_oversized_catalog_fails_before_llm():
    from app.context.budget import with_tool_prefix
    with pytest.raises(ContextCapacityError):
        with_tool_prefix(ContextConfig(window=18000,output=2000,user_input=2000,steps=3,tool_result=1200,rerank_k=5),
            [SystemMessage(content='前缀'*20000)],[])
