import json

from langchain_core.messages import AIMessage, ToolMessage

from agent.evidence import (
    MAX_STORED_EVIDENCE_CHARS,
    MAX_STORED_EVIDENCE_ITEMS,
    create_evidence,
    merge_evidence,
)
from agent.graph import record_tools_node


EVIDENCE_FIELDS = {
    "evidence_id",
    "docid",
    "url",
    "text",
    "source_tool",
    "query",
    "start_offset",
    "end_offset",
    "score",
    "rank",
}


def make_state(messages, evidence=None):
    return {
        "messages": messages,
        "tool_calls": 0,
        "active_queries": [],
        "visited_docs": [],
        "evidence": evidence or [],
        "trajectory": [],
    }


def test_create_evidence_has_stable_uniform_schema():
    kwargs = {
        "docid": "86190",
        "url": "https://example.com",
        "text": " established: 1996 ",
        "source_tool": "search_documents",
        "query": "Queen Arwa University established",
        "score": -27.98,
        "rank": 1,
    }

    first = create_evidence(**kwargs)
    second = create_evidence(**kwargs)

    assert first is not None
    assert set(first) == EVIDENCE_FIELDS
    assert first["text"] == "established: 1996"
    assert first["evidence_id"] == second["evidence_id"]


def test_merge_evidence_deduplicates_and_applies_bounds():
    additions = [
        create_evidence(
            docid=str(index),
            url=None,
            text=(f"evidence {index} " + "x" * 3000),
            source_tool="search_documents",
        )
        for index in range(MAX_STORED_EVIDENCE_ITEMS + 5)
    ]
    additions.append(additions[-1])

    merged = merge_evidence([], additions)

    assert len(merged) == MAX_STORED_EVIDENCE_ITEMS
    assert all(
        len(item["text"]) <= MAX_STORED_EVIDENCE_CHARS
        for item in merged
    )
    assert len({item["evidence_id"] for item in merged}) == len(merged)


def test_search_results_are_recorded_as_evidence_and_deduplicated():
    tool_call = {
        "name": "search_documents",
        "args": {"query": "Queen Arwa University established"},
        "id": "search-1",
        "type": "tool_call",
    }
    result = [{
        "rank": 1,
        "docid": "86190",
        "score": -27.98,
        "snippet": "Queen Arwa University ... established: 1996",
        "url": "https://en.wikipedia.org/wiki/Queen_Arwa_University",
    }]
    messages = [
        AIMessage(content="", tool_calls=[tool_call]),
        ToolMessage(
            content=json.dumps(result),
            tool_call_id="search-1",
            name="search_documents",
            status="success",
        ),
    ]

    first = record_tools_node(make_state(messages))
    second = record_tools_node(
        make_state(messages, evidence=first["evidence"])
    )

    assert len(first["evidence"]) == 1
    assert len(second["evidence"]) == 1
    evidence = first["evidence"][0]
    assert set(evidence) == EVIDENCE_FIELDS
    assert evidence["source_tool"] == "search_documents"
    assert evidence["query"] == "Queen Arwa University established"
    assert evidence["score"] == -27.98
    assert evidence["rank"] == 1


def test_document_tool_results_use_the_same_evidence_schema():
    calls = [
        {
            "name": "open_document",
            "args": {"docid": "5412"},
            "id": "open-1",
            "type": "tool_call",
        },
        {
            "name": "find_in_document",
            "args": {"docid": "5412", "keyword": "Cultural Week"},
            "id": "find-1",
            "type": "tool_call",
        },
    ]
    messages = [
        AIMessage(content="", tool_calls=calls),
        ToolMessage(
            content=json.dumps({
                "docid": "5412",
                "url": "https://example.com/5412",
                "content": "Queen Arwa University held the event.",
                "start_offset": 0,
                "end_offset": 38,
            }),
            tool_call_id="open-1",
            name="open_document",
            status="success",
        ),
        ToolMessage(
            content=json.dumps({
                "docid": "5412",
                "url": "https://example.com/5412",
                "matches": [{
                    "snippet": "Fifth Cultural Week on May 9-11 2002",
                    "start_offset": 300,
                    "end_offset": 420,
                }],
            }),
            tool_call_id="find-1",
            name="find_in_document",
            status="success",
        ),
    ]

    update = record_tools_node(make_state(messages))

    assert len(update["evidence"]) == 2
    assert all(set(item) == EVIDENCE_FIELDS for item in update["evidence"])
    by_tool = {
        item["source_tool"]: item
        for item in update["evidence"]
    }
    assert by_tool["open_document"]["start_offset"] == 0
    assert by_tool["find_in_document"]["query"] == "Cultural Week"
