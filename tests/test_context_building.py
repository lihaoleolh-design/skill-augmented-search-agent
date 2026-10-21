from langchain_core.messages import HumanMessage, SystemMessage

from agent.graph import build_messages
from agent.search_progress import initial_search_progress


def make_state(trajectory):
    snippet = "Queen Arwa University was established in 1996."
    return {
        "question": "Which university was established in which year?",
        "search_round": 1,
        "tool_calls": 1,
        "active_queries": ["Queen Arwa University established"],
        "visited_docs": [],
        "evidence": [{
            "evidence_id": "ev-1",
            "docid": "86190",
            "url": "https://example.com",
            "text": snippet,
            "source_tool": "search_documents",
            "query": "Queen Arwa University established",
            "score": -20.0,
            "rank": 1,
            "start_offset": None,
            "end_offset": None,
        }],
        "trajectory": trajectory,
        "required_facts": [],
        "skill_instructions": "",
        "empty_response_count": 0,
        "answer_validation": {},
        "search_progress": initial_search_progress(),
    }


def test_follow_up_context_keeps_system_message_before_user_message():
    messages = build_messages(make_state([]))

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "RECORDED SEARCH STATE" in messages[0].content


def test_successful_tool_snippet_is_not_duplicated_in_context():
    snippet = "Queen Arwa University was established in 1996."
    trajectory = [{
        "type": "tool_result",
        "tool": "search_documents",
        "status": "success",
        "output": [{
            "docid": "86190",
            "snippet": snippet,
            "rank": 1,
            "score": -20.0,
        }],
    }]

    messages = build_messages(make_state(trajectory))

    assert messages[0].content.count(snippet) == 1


def test_tool_errors_remain_available_in_recent_context():
    trajectory = [{
        "type": "tool_result",
        "tool": "open_document",
        "status": "error",
        "output": {"error": "document_not_found"},
    }]

    messages = build_messages(make_state(trajectory))

    assert "document_not_found" in messages[0].content
