from app.config import settings
from app.core.llm import get_chat_model

def test_factory_points_to_chat_upstream():
    m = get_chat_model()
    assert m.model_name == settings.chat_model      # 跟随配置,不硬编码模型名
    assert str(m.openai_api_base) == settings.chat_base_url

def test_factory_streaming_flag():
    assert get_chat_model(streaming=True).streaming is True

def test_disable_thinking_sets_extra_body():
    m = get_chat_model(disable_thinking=True)
    assert m.extra_body == {"thinking": {"type": "disabled"}}

def test_default_has_no_extra_body():
    assert get_chat_model().extra_body is None


def test_judge_can_set_zero_temperature_without_changing_chat_default():
    assert get_chat_model(temperature=0).temperature == 0
    assert get_chat_model().temperature == 0.3
