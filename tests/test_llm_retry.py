import pytest
from langchain_core.messages import HumanMessage

import agent.graph as graph_module
from agent.llm import is_retryable_llm_error


class ErrorModel:
    def __init__(self, error):
        self.error = error

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        raise self.error


class StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def make_state(**updates):
    state = {
        "question": "What is the answer?",
        "messages": [HumanMessage(content="What is the answer?")],
        "search_round": 0,
        "tool_calls": 0,
        "total_tokens": 0,
        "token_usage_complete": True,
        "active_queries": [],
        "visited_docs": [],
        "evidence": [],
        "trajectory": [],
        "final_answer": "",
        "selected_skills": [],
        "pending_skill": None,
        "active_skill": None,
        "skill_instructions": "",
        "skill_reason": "",
        "empty_response_count": 0,
        "termination_reason": "",
        "answer_validation_attempts": 0,
        "answer_validation": {},
        "required_facts": [],
        "search_progress": {},
        "llm_error_count": 0,
        "llm_retry_pending": False,
        "last_llm_error": "",
    }
    state.update(updates)
    return state


def test_retryable_error_classification():
    assert is_retryable_llm_error(TimeoutError("slow"))
    assert is_retryable_llm_error(StatusError(429))
    assert is_retryable_llm_error(StatusError(503))
    assert not is_retryable_llm_error(StatusError(400))
    assert not is_retryable_llm_error(ValueError("bad request"))


def test_agent_node_turns_timeout_into_retry_state(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: ErrorModel(TimeoutError("request timed out")),
    )

    update = graph_module.agent_node(make_state())
    routed_state = make_state(**update)

    assert update["llm_error_count"] == 1
    assert update["llm_retry_pending"] is True
    assert update["termination_reason"] == "retrying_llm_error"
    assert update["trajectory"][-1]["type"] == "llm_error"
    assert graph_module.route_after_agent(routed_state) == "retry_llm"


def test_non_retryable_llm_error_is_raised(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: ErrorModel(ValueError("invalid model input")),
    )

    with pytest.raises(ValueError, match="invalid model input"):
        graph_module.agent_node(make_state())


def test_run_ends_safely_when_timeout_retries_are_exhausted(monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: ErrorModel(TimeoutError("request timed out")),
    )

    result = graph_module.run_agent("Give me an unsupported fact")

    errors = [
        event
        for event in result["trajectory"]
        if event["type"] == "llm_error"
    ]
    assert len(errors) == graph_module.settings.llm_max_retries + 1
    assert errors[-1]["will_retry"] is False
    assert result["final_answer"] == "Unknown"
    assert result["termination_reason"] == "llm_error_retries_exhausted"
