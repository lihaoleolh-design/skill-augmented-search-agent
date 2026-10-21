from agent.answer_validator import extract_required_facts, validate_answer
from agent.graph import (
    END,
    evaluate_search_progress_node,
    route_after_search_progress,
)
from agent.search_progress import (
    evaluate_search_progress,
    initial_search_progress,
)
from config import settings


QUESTION = (
    "Which university held the event, and in which year was it established?"
)
ANSWER = "Queen Arwa University, 1996"


def evidence(evidence_id, text, docid="86190"):
    return {
        "evidence_id": evidence_id,
        "docid": docid,
        "url": "https://example.com",
        "text": text,
        "source_tool": "search_documents",
        "query": "Queen Arwa University established",
        "start_offset": None,
        "end_offset": None,
        "score": -20.0,
        "rank": 1,
    }


def search_trajectory(docid="86190", snippet="result"):
    return [
        {"type": "llm_response", "round": 2},
        {
            "type": "tool_result",
            "tool": "search_documents",
            "status": "success",
            "output": [{"docid": docid, "snippet": snippet}],
        },
    ]


def partial_facts():
    requirements = extract_required_facts(QUESTION)
    result = validate_answer(
        ANSWER,
        [evidence("ev-1", "Queen Arwa University held the event.")],
        requirements,
    )
    return result["required_facts"]


def progress_state(**updates):
    first_evidence = evidence(
        "ev-1",
        "Queen Arwa University held the event.",
    )
    state = {
        "final_answer": ANSWER,
        "evidence": [first_evidence],
        "required_facts": partial_facts(),
        "search_progress": {
            **initial_search_progress(),
            "evaluation_count": 1,
            "seen_evidence_ids": ["ev-1"],
            "seen_docids": ["86190"],
            "last_search_docids": ["86190"],
        },
        "trajectory": search_trajectory(),
        "tool_calls": 2,
        "termination_reason": "",
    }
    state.update(updates)
    return state


def test_initial_new_evidence_counts_as_progress():
    item = evidence("ev-1", "Queen Arwa University held the event.")
    state = progress_state(
        final_answer="",
        evidence=[item],
        required_facts=extract_required_facts(QUESTION),
        search_progress=initial_search_progress(),
    )

    progress = evaluate_search_progress(state)

    assert progress["made_progress"] is True
    assert progress["reason"] == "initial_evidence_discovered"
    assert progress["new_evidence_count"] == 1
    assert progress["stalled_rounds"] == 0


def test_similar_results_can_add_missing_fact_evidence():
    complete = evidence(
        "ev-2",
        "Queen Arwa University was established in 1996.",
    )
    state = progress_state(
        evidence=[*progress_state()["evidence"], complete],
        trajectory=search_trajectory(
            snippet="Queen Arwa University was established in 1996."
        ),
    )

    progress = evaluate_search_progress(state)

    assert progress["document_overlap"] == 1.0
    assert progress["new_document_count"] == 0
    assert progress["new_evidence_count"] == 1
    assert progress["new_supported_fact_count"] == 1
    assert progress["evidence_sufficient"] is True
    assert progress["made_progress"] is True

    routed_state = {
        **state,
        "search_progress": {
            key: value
            for key, value in progress.items()
            if key != "required_facts"
        },
    }
    assert route_after_search_progress(routed_state) == "validate_answer"


def test_repeated_results_without_new_evidence_increment_stall():
    state = progress_state()

    progress = evaluate_search_progress(state)

    assert progress["document_overlap"] == 1.0
    assert progress["new_evidence_count"] == 0
    assert progress["made_progress"] is False
    assert progress["reason"] == "no_new_evidence"
    assert progress["stalled_rounds"] == 1


def test_new_but_irrelevant_evidence_is_not_effective_progress():
    unrelated = evidence(
        "ev-2",
        "An unrelated college was established in 1996.",
        docid="other",
    )
    state = progress_state(
        evidence=[*progress_state()["evidence"], unrelated],
        trajectory=search_trajectory(
            docid="other",
            snippet="An unrelated college was established in 1996.",
        ),
    )

    progress = evaluate_search_progress(state)

    assert progress["new_evidence_count"] == 1
    assert progress["new_document_count"] == 1
    assert progress["new_supported_fact_count"] == 0
    assert progress["made_progress"] is False
    assert progress["reason"] == "new_evidence_did_not_resolve_missing_facts"


def test_progress_node_ends_after_configured_stall_limit():
    previous = {
        **progress_state()["search_progress"],
        "stalled_rounds": settings.max_stalled_searches - 1,
    }
    state = progress_state(search_progress=previous)

    update = evaluate_search_progress_node(state)
    routed = {**state, **update}

    assert update["search_progress"]["stalled_rounds"] == settings.max_stalled_searches
    assert update["final_answer"] == "Unknown"
    assert update["termination_reason"] == "search_stalled"
    assert update["trajectory"][-1]["type"] == "search_progress"
    assert route_after_search_progress(routed) == END
