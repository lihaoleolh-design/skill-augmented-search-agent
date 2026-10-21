from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class SearchAgentState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    search_round: int
    tool_calls: int
    total_tokens: int
    token_usage_complete: bool
    active_queries: list[str]
    visited_docs: list[str]
    evidence: list[dict[str, Any]]
    trajectory: list[dict[str, Any]]
    final_answer: str
    selected_skills: list[str]
    pending_skill: str | None
    active_skill: str | None
    skill_instructions: str
    skill_reason: str
    empty_response_count: int
    termination_reason: str
    answer_validation_attempts: int
    answer_validation: dict[str, Any]
    required_facts: list[dict[str, Any]]
    search_progress: dict[str, Any]
    llm_error_count: int
    llm_retry_pending: bool
    last_llm_error: str
    force_tool_call: bool
    last_validated_answer: str
    repeated_candidate_count: int
