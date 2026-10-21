import pytest

from agent.skill_loader import discover_skills, load_skill
from agent.skill_router import needs_recovery, select_skill


def make_state(question, **updates):
    state = {
        "question": question,
        "selected_skills": [],
        "search_round": 0,
        "tool_calls": 0,
        "trajectory": [],
        "active_queries": [],
    }
    state.update(updates)
    return state


def test_discovers_expected_skills():
    discovered = {
        item["name"]
        for item in discover_skills()
    }

    assert {
        "entity-disambiguation",
        "multi-hop-search",
        "search-recovery",
    }.issubset(discovered)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Which university held its Fifth Cultural Week in 2002, "
            "and when was it founded?",
            "multi-hop-search",
        ),
        (
            "Which person named John Smith was president rather than "
            "the scientist?",
            "entity-disambiguation",
        ),
        (
            "On which dates did Queen Arwa University hold its Fifth "
            "Cultural Week?",
            None,
        ),
    ],
)
def test_selects_skill_for_question_type(question, expected):
    selected, _reason = select_skill(make_state(question))

    assert selected == expected

    if selected:
        assert "#" in load_skill(selected)


def test_selects_search_recovery_after_empty_search():
    state = make_state(
        "Find a rare item",
        search_round=1,
        tool_calls=1,
        trajectory=[
            {"type": "llm_response", "round": 1},
            {
                "type": "tool_result",
                "tool": "search_documents",
                "status": "success",
                "output": [],
            },
        ],
    )

    assert needs_recovery(state)
    assert select_skill(state)[0] == "search-recovery"


def test_search_recovery_is_only_selected_once():
    state = make_state(
        "Find a rare item",
        selected_skills=["search-recovery"],
        search_round=1,
        tool_calls=1,
        trajectory=[
            {"type": "llm_response", "round": 1},
            {
                "type": "tool_result",
                "tool": "search_documents",
                "status": "success",
                "output": [],
            },
        ],
    )

    assert not needs_recovery(state)
    assert select_skill(state)[0] is None
