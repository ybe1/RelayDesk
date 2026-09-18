from app.config import Settings

import pytest
from pydantic import ValidationError

def test_账号相关的三项必填(monkeypatch):
    # 这三项跟着账号走,给默认值等于替人猜:猜错不在启动时报错,而是第一次调上游时
    # 抛一个看不懂的 401
    for k in ("CHAT_MODEL", "CHAT_BASE_URL", "CHAT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

def test_settings_env_override(monkeypatch):
    for k, v in {"CHAT_MODEL": "m", "CHAT_BASE_URL": "u", "CHAT_API_KEY": "k"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("TOKEN_BUDGET", "500")
    assert Settings(_env_file=None).token_budget == 500
