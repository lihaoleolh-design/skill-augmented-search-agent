from agent.llm import get_llm
from config import settings


def test_thinking_is_explicitly_disabled_for_llm_requests():
    get_llm.cache_clear()
    model = get_llm()

    assert settings.llm_enable_thinking is False
    assert model.extra_body == {
        "enable_thinking": False
    }
