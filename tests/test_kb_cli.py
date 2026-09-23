from contextlib import contextmanager
from types import SimpleNamespace


def test_preview_is_read_only_and_reports_all_counters(monkeypatch, capsys):
    from scripts import kb

    @contextmanager
    def forbidden_lock(_engine):
        raise AssertionError("preview must not acquire writer lock")
        yield

    monkeypatch.setattr(kb, "preview", lambda **_: {"imported": 0, "skipped": 0, "indexed": 0, "pending": 2, "failed": 0})

    assert kb.main(["preview"], lock=forbidden_lock) == 0
    assert '"pending": 2' in capsys.readouterr().out


def test_vectorize_holds_writer_lock_and_exposes_crash_hook_only_for_cli(monkeypatch):
    from scripts import kb

    events = []

    @contextmanager
    def lock(_engine):
        events.append("locked")
        yield
        events.append("released")

    class Store:
        def ensure_collection(self):
            events.append("collection")

    monkeypatch.setattr(kb, "MilvusStore", lambda **_: Store())
    monkeypatch.setattr(kb, "EmbeddingClient", lambda **_: object())
    monkeypatch.setattr(kb, "settings", SimpleNamespace(
        milvus_uri="http://milvus.test", embed_base_url="http://embed.test",
        embed_api_key="test", embed_model="BAAI/bge-m3",
    ))
    monkeypatch.setattr(kb, "index_pending", lambda *args, **kwargs: {"indexed": 1, "pending": 0, "failed": 0})

    fake_session_factory = lambda: None
    assert kb.main(["vectorize"], lock=lock, database_engine=object(), session_factory=fake_session_factory) == 0
    assert events == ["locked", "collection", "released"]

    exit_codes = []
    def crashable_index(*args, **kwargs):
        kwargs["after_upsert"]()
        return {"indexed": 1, "pending": 0, "failed": 0}

    monkeypatch.setattr(kb, "index_pending", crashable_index)
    monkeypatch.setattr(kb.os, "_exit", lambda code: exit_codes.append(code))
    assert kb.main(["vectorize", "--crash-after-upsert"], lock=lock, database_engine=object(), session_factory=fake_session_factory) == 0
    assert exit_codes == [86]
