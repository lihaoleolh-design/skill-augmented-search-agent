from typing import Any

from agent.answer_validator import validate_answer


def initial_search_progress() -> dict[str, Any]:
    return {
        "evaluation_count": 0,
        "seen_evidence_ids": [],
        "seen_docids": [],
        "last_search_docids": [],
        "new_evidence_count": 0,
        "new_document_count": 0,
        "new_supported_fact_count": 0,
        "document_overlap": None,
        "made_progress": False,
        "stalled_rounds": 0,
        "evidence_sufficient": False,
        "reason": "not_evaluated",
    }


def _latest_tool_results(trajectory: list[dict[str, Any]]):
    last_llm_index = next(
        (
            index
            for index in range(len(trajectory) - 1, -1, -1)
            if trajectory[index].get("type") == "llm_response"
        ),
        -1,
    )
    return [
        event
        for event in trajectory[last_llm_index + 1:]
        if event.get("type") == "tool_result"
    ]


def _search_docids(events: list[dict[str, Any]]) -> list[str]:
    docids = []
    for event in events:
        if event.get("tool") != "search_documents":
            continue
        output = event.get("output")
        if not isinstance(output, list):
            continue
        for item in output:
            if isinstance(item, dict) and item.get("docid") is not None:
                docid = str(item["docid"])
                if docid not in docids:
                    docids.append(docid)
    return docids


def _jaccard(previous: list[str], current: list[str]) -> float | None:
    if not previous or not current:
        return None
    previous_set = set(previous)
    current_set = set(current)
    return round(
        len(previous_set & current_set) /
        len(previous_set | current_set),
        4,
    )


def evaluate_search_progress(state: dict[str, Any]) -> dict[str, Any]:
    """Measure whether the latest tool batch reduced unresolved work."""
    previous = state.get("search_progress") or initial_search_progress()
    events = _latest_tool_results(state.get("trajectory", []))
    current_search_docids = _search_docids(events)

    evidence = state.get("evidence", [])
    current_evidence_ids = [
        item["evidence_id"]
        for item in evidence
        if item.get("evidence_id")
    ]
    current_docids = list(dict.fromkeys(
        str(item["docid"])
        for item in evidence
        if item.get("docid") is not None
    ))
    new_evidence_ids = set(current_evidence_ids) - set(
        previous.get("seen_evidence_ids", [])
    )
    new_docids = set(current_docids) - set(
        previous.get("seen_docids", [])
    )

    required_facts = state.get("required_facts", [])
    previous_supported = {
        fact.get("fact_id")
        for fact in required_facts
        if fact.get("status") == "supported"
    }
    validation = None
    answer = state.get("final_answer", "")
    if answer and answer.strip().lower() != "unknown":
        validation = validate_answer(
            answer,
            evidence,
            required_facts,
        )
        evaluated_facts = validation["required_facts"]
    else:
        evaluated_facts = required_facts

    now_supported = {
        fact.get("fact_id")
        for fact in evaluated_facts
        if fact.get("status") == "supported"
    }
    new_supported_count = len(now_supported - previous_supported)
    evidence_sufficient = bool(
        validation and validation.get("evidence_complete")
    )

    has_candidate_facts = any(
        fact.get("values")
        for fact in required_facts
    )
    if evidence_sufficient:
        made_progress = True
        reason = "evidence_complete"
    elif new_supported_count > 0:
        made_progress = True
        reason = "required_fact_supported"
    elif not has_candidate_facts and new_evidence_ids:
        made_progress = True
        reason = "initial_evidence_discovered"
    elif new_evidence_ids:
        made_progress = False
        reason = "new_evidence_did_not_resolve_missing_facts"
    else:
        made_progress = False
        reason = "no_new_evidence"

    stalled_rounds = (
        0
        if made_progress
        else previous.get("stalled_rounds", 0) + 1
    )

    return {
        "evaluation_count": previous.get("evaluation_count", 0) + 1,
        "seen_evidence_ids": current_evidence_ids,
        "seen_docids": current_docids,
        "last_search_docids": (
            current_search_docids
            if current_search_docids
            else previous.get("last_search_docids", [])
        ),
        "new_evidence_count": len(new_evidence_ids),
        "new_document_count": len(new_docids),
        "new_supported_fact_count": new_supported_count,
        "document_overlap": _jaccard(
            previous.get("last_search_docids", []),
            current_search_docids,
        ),
        "made_progress": made_progress,
        "stalled_rounds": stalled_rounds,
        "evidence_sufficient": evidence_sufficient,
        "reason": reason,
        "required_facts": evaluated_facts,
    }
