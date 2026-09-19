import json
from types import SimpleNamespace

from app.tools.business import build_tools


def test_mock_queries_return_requested_identity_and_mock_marker():
    tools = {tool.name: tool for tool in build_tools(lambda: None, 42)}
    for name, args, key, value in (
        ("query_order", {"order_id": "1001"}, "order_id", "1001"),
        ("query_product", {"product_name": "猫粮"}, "product_name", "猫粮"),
        ("query_logistics", {"order_id": "1001"}, "order_id", "1001"),
    ):
        result = json.loads(tools[name].invoke(args))
        assert result["mock"] is True
        assert result[key] == value


def test_ticket_schema_excludes_model_supplied_conversation_id():
    tools = {tool.name: tool for tool in build_tools(lambda: None, 42)}
    schema = tools["create_ticket"].args_schema.model_json_schema()
    assert "conversation_id" not in schema["properties"]
    assert set(tools) == {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}


def test_faq_returns_repository_hit_and_miss(monkeypatch):
    from app.repositories import faq

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def search(_session, keyword):
        if keyword == "退货政策":
            return [SimpleNamespace(question="退货政策是什么", answer="七天内可申请", category="售后")]
        return []

    monkeypatch.setattr(faq, "search_faq", search)
    query = {item.name: item for item in build_tools(Session, 42)}["query_faq"]
    assert json.loads(query.invoke({"keyword": "退货政策"}))["matches"][0]["answer"] == "七天内可申请"
    assert json.loads(query.invoke({"keyword": "邮费"}))["matches"] == []


def test_ticket_uses_server_conversation_and_commits(monkeypatch):
    from app.repositories import tickets

    calls = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def commit(self):
            calls.append("commit")

        def rollback(self):
            calls.append("rollback")

    def save(_session, conversation_id, description, ticket_type):
        calls.append((conversation_id, description, ticket_type))
        return SimpleNamespace(ticket_no="T0001")

    monkeypatch.setattr(tickets, "create_ticket", save)
    ticket = {item.name: item for item in build_tools(Session, 42)}["create_ticket"]
    result = json.loads(ticket.invoke({"description": "请人工处理", "ticket_type": "售后"}))
    assert result["ticket_no"] == "T0001"
    assert calls == [(42, "请人工处理", "售后"), "commit"]
