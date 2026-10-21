
import json
from pathlib import Path
from datetime import datetime

from agent.graph import run_agent
from agent.hooks import ConsoleTraceHook


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "trajectories"


def save_trajectory(question: str, result: dict) -> Path:
    """
    Save complete agent execution information.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trajectory = result.get("trajectory", [])

    skill_events = [
        event
        for event in trajectory
        if event.get("type") == "skill_loaded"
    ]

    output = {
        "question": question,
        "answer": result.get("final_answer", ""),

        # Agent execution statistics
        "statistics": {
            "llm_rounds": result.get("search_round", 0),
            "tool_calls": result.get("tool_calls", 0),
            "total_tokens": result.get("total_tokens", 0),
            "token_usage_complete": result.get(
                "token_usage_complete", False
            ),
            "evidence_count": len(
                result.get("evidence", [])
            ),
            "skill_load_count": len(skill_events),
            "termination_reason": result.get(
                "termination_reason", ""
            )
        },

        # Skill information
        "skills": {
            "selected_skills": result.get(
                "selected_skills", []
            ),
            "active_skill": result.get("active_skill"),
            "load_events": skill_events
        },

        # Final answer validation
        "validation": {
            "attempts": result.get(
                "answer_validation_attempts", 0
            ),
            "result": result.get(
                "answer_validation", {}
            ),
            "termination_reason": result.get(
                "termination_reason", ""
            )
        },

        "search_progress": result.get(
            "search_progress", {}
        ),

        # Retrieval information
        "retrieval": {
            "search_queries": result.get(
                "active_queries", []
            ),
            "visited_documents": result.get(
                "visited_docs", []
            ),
            "evidence": result.get("evidence", [])
        },

        # Full execution trajectory
        "trajectory": trajectory
    }

    filename = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )

    output_path = OUTPUT_DIR / filename

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    return output_path


def print_agent_statistics(result: dict):
    """
    Print agent and skill execution statistics.
    """

    trajectory = result.get("trajectory", [])

    skill_events = [
        event
        for event in trajectory
        if event.get("type") == "skill_loaded"
    ]

    print("\nAgent Statistics")
    print("-" * 60)

    print(
        "LLM rounds:",
        result.get("search_round", 0)
    )

    print(
        "Tool calls:",
        result.get("tool_calls", 0)
    )

    print(
        "Total tokens:",
        result.get("total_tokens", 0)
    )

    print(
        "Token usage complete:",
        result.get("token_usage_complete", False)
    )

    print(
        "Termination reason:",
        result.get("termination_reason", "")
    )

    print(
        "Visited documents:",
        result.get("visited_docs", [])
    )

    print(
        "Search queries:",
        result.get("active_queries", [])
    )

    print(
        "Evidence count:",
        len(result.get("evidence", []))
    )

    print("\nSkill Statistics")
    print("-" * 60)

    print(
        "Selected Skills:",
        result.get("selected_skills", [])
    )

    print(
        "Active Skill:",
        result.get("active_skill")
    )

    print(
        "Skill Load Events:",
        len(skill_events)
    )

    for index, event in enumerate(skill_events, start=1):
        print(
            f"  {index}. "
            f"{event.get('skill', 'unknown')}"
        )

        if event.get("reason"):
            print(
                f"     Reason: {event['reason']}"
            )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)

    trace_hook = ConsoleTraceHook()
    print("COMP5423 - Skill-Augmented Search Agent")
    print("=" * 60)

    while True:
        try:
            question = input(
                "\nEnter question (q to quit): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() in ("q", "quit", "exit"):
            break

        if not question:
            continue

        try:
            result = run_agent(
                question,
                hooks=trace_hook,
            )

            print("\n" + "=" * 60)
            print("Final Answer")
            print("=" * 60)
            print(
                result.get("final_answer", "")
            )

            print_agent_statistics(result)

            output_path = save_trajectory(
                question,
                result
            )

            print(
                "\nTrajectory saved:",
                output_path
            )

        except Exception as e:
            print(
                f"\nAgent execution failed: "
                f"{type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()
