from functools import lru_cache
from langchain_openai import ChatOpenAI

from config import settings


def is_retryable_llm_error(error: Exception) -> bool:
    """Return whether an LLM failure is likely transient."""
    name = type(error).__name__.lower()
    if any(term in name for term in (
        "timeout", "connection", "ratelimit", "rate_limit"
    )):
        return True

    status_code = getattr(error, "status_code", None)
    return status_code == 429 or (
        isinstance(status_code, int) and status_code >= 500
    )


@lru_cache
def get_llm():
    if not all([
        settings.llm_api_key,
        settings.llm_base_url,
        settings.llm_model
    ]):
        raise ValueError(
            "Missing LLM_API_KEY, LLM_BASE_URL "
            "or LLM_MODEL in .env"
        )

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        extra_body={
            "enable_thinking": settings.llm_enable_thinking
        },
        # Retries are handled explicitly by the Agent graph so they are
        # observable through hooks and have a clear terminal state.
        max_retries=0
    )
