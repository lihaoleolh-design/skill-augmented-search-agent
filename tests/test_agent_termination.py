from langchain_core.messages import AIMessage, HumanMessage

import agent.graph as graph_module


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
    }
    state.update(updates)
    return state


class FakeModel:
    def __init__(self, response):
        self.response = response

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return self.response


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return next(self.responses)


def test_extract_response_text_handles_provider_block_content():
    content = [
        {"type": "text", "text": " Queen Arwa University "},
        {"type": "metadata", "value": "ignored"},
    ]

    assert graph_module.extract_response_text(content) == "Queen Arwa University"
    assert graph_module.extract_response_text([]) == ""


def test_first_empty_response_is_retried(monkeypatch):
    response = AIMessage(content=[])
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: FakeModel(response),
    )

    update = graph_module.agent_node(make_state())
    routed_state = make_state(
        messages=[response],
        empty_response_count=update["empty_response_count"],
    )

    assert update["final_answer"] == ""
    assert update["termination_reason"] == "retrying_empty_response"
    assert graph_module.route_after_agent(routed_state) == "retry_empty"


def test_repeated_empty_response_falls_back_to_unknown(monkeypatch):
    response = AIMessage(content="")
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: FakeModel(response),
    )

    update = graph_module.agent_node(
        make_state(empty_response_count=1)
    )
    routed_state = make_state(
        messages=[response],
        empty_response_count=update["empty_response_count"],
    )

    assert update["final_answer"] == "Unknown"
    assert update["termination_reason"] == "llm_empty_response"
    assert graph_module.route_after_agent(routed_state) == "validate_answer"


def test_non_empty_response_completes_normally(monkeypatch):
    response = AIMessage(content="1996")
    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: FakeModel(response),
    )

    update = graph_module.agent_node(make_state())

    assert update["final_answer"] == "1996"
    assert update["empty_response_count"] == 0
    assert update["termination_reason"] == "completed"


def test_graph_retries_once_then_returns_unknown(monkeypatch):
    model = SequenceModel([
        AIMessage(content=[]),
        AIMessage(content=""),
    ])
    monkeypatch.setattr(graph_module, "get_llm", lambda: model)

    result = graph_module.run_agent("Give me an unsupported fact")

    assert result["search_round"] == 2
    assert result["final_answer"] == "Unknown"
    assert result["termination_reason"] == "llm_empty_response"
