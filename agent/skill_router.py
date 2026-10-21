
"""
Low-cost, deterministic skill selection.

Route once initially and again when retrieval fails.
"""

import re
from datetime import datetime, timezone
from typing import Any, TypedDict

from config import settings
from agent.state import SearchAgentState
from agent.skill_loader import discover_skills, load_skill


class SkillRouterUpdate(TypedDict, total=False):
    pending_skill: str | None
    skill_reason: str


class SkillLoaderUpdate(TypedDict, total=False):
    pending_skill: str | None
    active_skill: str | None
    skill_instructions: str
    selected_skills: list[str]
    trajectory: list[dict[str, Any]]


# =========================================================
# Question Classification
# =========================================================

MULTI_HOP_PATTERNS = [
    r"\band (?:in )?which\b",
    r"\band when\b",
    r"\band where\b",
    r"\bwhose\b",
    r"\bthe person who\b",
    r"\bthe university that\b",
    r"\bwho.*\band\b.*\bwhere\b",
]

ENTITY_PATTERNS = [
    r"\bwhich of\b",
    r"\bwhich one\b",
    r"\bsame name\b",
    r"\bnot to be confused\b",
    r"\bdisambiguat",
    r"\bdifferent people\b",
    r"\bwhich person named\b",
    r"\bwhich university named\b",
    r"\brather than\b",
]


def classify_question(question: str) -> str | None:
    text = question.lower()

    if any(
        re.search(pattern, text)
        for pattern in ENTITY_PATTERNS
    ):
        return "entity-disambiguation"

    if any(
        re.search(pattern, text)
        for pattern in MULTI_HOP_PATTERNS
    ):
        return "multi-hop-search"

    return None


# =========================================================
# Search Recovery
# =========================================================

def needs_recovery(state: SearchAgentState) -> bool:
    """
    Trigger recovery if every search in the latest
    tool batch failed or returned no results.

    Recovery is only activated once.
    """

    if "search-recovery" in state["selected_skills"]:
        return False

    if state["tool_calls"] >= settings.max_tool_calls:
        return False

    events = state["trajectory"]

    last_llm_index = next(
        (
            i
            for i in range(len(events) - 1, -1, -1)
            if events[i].get("type") == "llm_response"
        ),
        -1
    )

    if last_llm_index == -1:
        return False

    results = [
        event
        for event in events[last_llm_index + 1:]
        if (
            event.get("type") == "tool_result"
            and event.get("tool") == "search_documents"
        )
    ]

    if not results:
        return False

    for event in results:
        if event.get("status") == "error":
            continue

        output = event.get("output")

        # Search tool directly returned a list.
        if isinstance(output, list) and output:
            return False

        # Alternative search tool response format.
        if isinstance(output, dict):
            if output.get("results"):
                return False

    return True


# =========================================================
# Skill Selection
# =========================================================

def select_skill(
    state: SearchAgentState
) -> tuple[str | None, str]:
    """
    Return selected skill name and selection reason.
    """

    available = {
        item["name"]
        for item in discover_skills()
    }

    selected = state["selected_skills"]

    # Priority 1: recover failed retrieval.
    if needs_recovery(state):
        if "search-recovery" in available:
            return (
                "search-recovery",
                "unsuccessful_retrieval"
            )

    # Priority 2: classify the initial question.
    if (
        not selected
        and state["search_round"] == 0
    ):
        proposed = classify_question(state["question"])

        if proposed in available:
            return (
                proposed,
                "question_classification"
            )

    return None, "no_skill_needed"


# =========================================================
# LangGraph Nodes
# =========================================================

def skill_router_node(
    state: SearchAgentState
) -> SkillRouterUpdate:
    """
    Select an appropriate skill and record the reason.
    """

    skill, reason = select_skill(state)

    if skill is None or skill in state["selected_skills"]:
        return {
            "pending_skill": None,
            "skill_reason": reason
        }

    return {
        "pending_skill": skill,
        "skill_reason": reason
    }


def skill_loader_node(
    state: SearchAgentState
) -> SkillLoaderUpdate:
    """
    Load the selected SKILL.md instructions.
    """

    skill = state["pending_skill"]

    if not skill:
        return {}

    instructions = load_skill(skill)

    event: dict[str, Any] = {
        "type": "skill_loaded",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "skill": skill,
        "reason": state["skill_reason"]
    }

    return {
        "pending_skill": None,
        "active_skill": skill,
        "skill_instructions": instructions,
        "selected_skills": [
            *state["selected_skills"],
            skill
        ],
        "trajectory": [
            *state["trajectory"],
            event
        ]
    }
