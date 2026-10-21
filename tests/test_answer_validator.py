from agent.answer_validator import (
    canonical_answer,
    extract_required_facts,
    validate_answer,
)
from config import settings
from agent.graph import (
    END,
    route_after_answer_validation,
    validate_answer_node,
)


def evidence(text, evidence_id="ev-1"):
    return {
        "evidence_id": evidence_id,
        "docid": "86190",
        "url": "https://example.com",
        "text": text,
        "source_tool": "search_documents",
        "query": "Queen Arwa University established",
        "start_offset": None,
        "end_offset": None,
        "score": -20.0,
        "rank": 1,
    }


def validation_state(**updates):
    state = {
        "final_answer": "1996",
        "evidence": [],
        "answer_validation_attempts": 0,
        "answer_validation": {},
        "search_round": 1,
        "tool_calls": 0,
        "trajectory": [],
        "termination_reason": "",
        "required_facts": extract_required_facts(
            "In which year was it established?"
        ),
    }
    state.update(updates)
    return state


def test_exact_answer_is_supported():
    result = validate_answer(
        "Queen Arwa University",
        [evidence("The institution was Queen Arwa University.")],
    )

    assert result["valid"] is True
    assert result["reason"] == "supported"
    assert result["supporting_evidence_ids"] == ["ev-1"]


def test_multi_part_answer_can_be_supported_across_evidence():
    required_facts = extract_required_facts(
        "Which university held the event, and in which year was that "
        "university established?"
    )
    result = validate_answer(
        "Queen Arwa University, 1996",
        [
            evidence("Queen Arwa University held the event.", "ev-1"),
            evidence(
                "Queen Arwa University was established in 1996.",
                "ev-2",
            ),
        ],
        required_facts,
    )

    assert result["valid"] is True
    assert result["support_ratio"] == 1.0
    assert result["supporting_evidence_ids"] == ["ev-1", "ev-2"]


def test_verbose_answer_extracts_relationship_year_and_canonical_values():
    required_facts = extract_required_facts(
        "Which university held the event, and in which year was it established?"
    )
    answer = """I have sufficient evidence.

From document 5412: the Queen Arwa University held the event in 2002.
From Wikipedia: established: 1996.
The organization is Queen Arwa University, established in 1996.

Queen Arwa University
1996"""
    result = validate_answer(
        answer,
        [
            evidence("Queen Arwa University held the event in 2002.", "ev-1"),
            evidence(
                "Queen Arwa University was established in 1996.",
                "ev-2",
            ),
        ],
        required_facts,
    )

    facts = {fact["type"]: fact for fact in result["required_facts"]}
    assert result["valid"] is True
    assert facts["organization"]["values"] == ["Queen Arwa University"]
    assert facts["year"]["values"] == ["1996"]
    assert canonical_answer(result, answer) == "Queen Arwa University\n1996"


def test_person_extraction_prefers_titled_person_over_organization():
    required_facts = extract_required_facts(
        "Which university organized the event, and who was the chairman?"
    )
    answer = """Queen Arwa University organized the event.
Dr. Abdulaziz Abdulghani was the chairman.

Queen Arwa University
Dr. Abdulaziz Abdulghani"""
    result = validate_answer(
        answer,
        [
            evidence("Queen Arwa University organized the event.", "ev-1"),
            evidence(
                "Dr. Abdulaziz Abdulghani, the Shura Council Chairman.",
                "ev-2",
            ),
        ],
        required_facts,
    )

    facts = {fact["type"]: fact for fact in result["required_facts"]}
    assert result["valid"] is True
    assert facts["person"]["values"] == ["Dr. Abdulaziz Abdulghani"]


def test_unsupported_answer_reports_missing_terms():
    required_facts = extract_required_facts(
        "Which university held the event, and in which year was it established?"
    )
    result = validate_answer(
        "University of Example, 1988",
        [evidence("Queen Arwa University was established in 1996.")],
        required_facts,
    )

    assert result["valid"] is False
    assert result["reason"] == "incomplete_evidence"
    assert "requested organization" in result["missing_terms"]
    assert "requested year" in result["missing_terms"]


def test_partial_multi_hop_evidence_does_not_pass():
    required_facts = extract_required_facts(
        "Which university held the event, and in which year was it established?"
    )
    result = validate_answer(
        "Queen Arwa University, 1996",
        [evidence("Queen Arwa University held the event.")],
        required_facts,
    )

    facts = {fact["type"]: fact for fact in result["required_facts"]}
    assert result["valid"] is False
    assert facts["organization"]["status"] == "supported"
    assert facts["year"]["status"] == "missing"


def test_unrelated_year_evidence_does_not_complete_entity_chain():
    required_facts = extract_required_facts(
        "Which university held the event, and in which year was it established?"
    )
    result = validate_answer(
        "Queen Arwa University, 1996",
        [
            evidence("Queen Arwa University held the event.", "ev-1"),
            evidence("An unrelated college was established in 1996.", "ev-2"),
        ],
        required_facts,
    )

    facts = {fact["type"]: fact for fact in result["required_facts"]}
    assert result["valid"] is False
    assert facts["year"]["status"] == "missing"


def test_answer_without_evidence_is_rejected():
    result = validate_answer("1996", [])

    assert result["valid"] is False
    assert result["reason"] == "incomplete_evidence"


def test_unknown_is_incomplete_while_search_budget_remains():
    result = validate_answer("Unknown", [])

    assert result["valid"] is False
    assert result["reason"] == "insufficient_evidence"


def test_validation_node_retries_incomplete_evidence():
    update = validate_answer_node(validation_state())
    routed = validation_state(**update)

    assert update["final_answer"] == "1996"
    assert update["termination_reason"] == "retrying_answer_validation"
    assert update["force_tool_call"] is True
    assert update["trajectory"][-1]["will_retry"] is True
    assert route_after_answer_validation(routed) == "agent"


def test_validation_attempt_count_does_not_stop_search_early():
    update = validate_answer_node(
        validation_state(answer_validation_attempts=5)
    )

    assert update["termination_reason"] == "retrying_answer_validation"
    assert update["trajectory"][-1]["will_retry"] is True


def test_unknown_retries_while_search_budget_remains():
    update = validate_answer_node(
        validation_state(final_answer="Unknown")
    )

    assert update["termination_reason"] == "retrying_answer_validation"
    assert update["trajectory"][-1]["will_retry"] is True


def test_validation_node_falls_back_when_search_rounds_are_exhausted():
    update = validate_answer_node(
        validation_state(search_round=settings.max_agent_steps)
    )
    routed = validation_state(**update)

    assert update["final_answer"] == "Unknown"
    assert update["termination_reason"] == "incomplete_evidence"
    assert update["trajectory"][-1]["will_retry"] is False
    assert route_after_answer_validation(routed) == END


def test_repeated_invalid_candidate_stops_without_using_all_rounds():
    candidate = "Unsupported Example University, 1988"
    normalized = " ".join(candidate.lower().split())
    update = validate_answer_node(
        validation_state(
            final_answer=candidate,
            last_validated_answer=normalized,
            repeated_candidate_count=settings.max_repeated_candidates - 1,
        )
    )

    assert update["final_answer"] == "Unknown"
    assert update["termination_reason"] == "repeated_invalid_candidate"
    assert update["force_tool_call"] is False


def test_supported_answer_completes_with_validation_event():
    update = validate_answer_node(
        validation_state(
            evidence=[evidence("Established in 1996.")]
        )
    )
    routed = validation_state(**update)

    assert update["final_answer"] == "1996"
    assert update["termination_reason"] == "completed"
    assert update["trajectory"][-1]["valid"] is True
    assert route_after_answer_validation(routed) == END
