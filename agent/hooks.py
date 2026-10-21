import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, TextIO

from langchain_core.messages import AIMessage, ToolMessage


GRAY = "\033[90m"
RESET = "\033[0m"


class AgentRunHook(Protocol):
    """Middleware interface for observing an Agent run."""

    def on_run_start(self, event: dict[str, Any]) -> None: ...

    def on_step_start(self, event: dict[str, Any]) -> None: ...

    def on_step(self, event: dict[str, Any]) -> None: ...

    def on_run_end(self, event: dict[str, Any]) -> None: ...

    def on_run_error(self, event: dict[str, Any]) -> None: ...


class HookManager:
    """Dispatch lifecycle events without coupling hooks to the graph."""

    def __init__(
        self,
        hooks: AgentRunHook | Iterable[AgentRunHook] | None = None,
    ):
        if hooks is None:
            self.hooks = []
        elif all(
            hasattr(hooks, method)
            for method in (
                "on_run_start", "on_step", "on_run_end", "on_run_error"
            )
        ):
            self.hooks = [hooks]
        else:
            self.hooks = list(hooks)

        self.run_id = uuid.uuid4().hex
        self._last_event_time = time.perf_counter()
        self._node_start_times: dict[str, list[float]] = {}

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _dispatch(self, method: str, event: dict[str, Any]) -> None:
        for hook in self.hooks:
            try:
                callback = getattr(hook, method, None)
                if callback is not None:
                    callback(event)
            except Exception:
                # Observability must never break the Agent itself.
                continue

    def run_start(
        self,
        question: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._last_event_time = time.perf_counter()
        self._dispatch("on_run_start", {
            "type": "run_start",
            "run_id": self.run_id,
            "timestamp": self._timestamp(),
            "question": question,
            "metadata": metadata or {},
        })

    def step(
        self,
        node: str,
        update: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        now = time.perf_counter()
        starts = self._node_start_times.get(node, [])
        started_at = starts.pop(0) if starts else self._last_event_time
        elapsed = now - started_at
        self._last_event_time = now
        self._dispatch("on_step", {
            "type": "step",
            "run_id": self.run_id,
            "timestamp": self._timestamp(),
            "node": node,
            "elapsed_seconds": elapsed,
            "update": update,
            "state": state,
        })

    def step_start(
        self,
        node: str,
        task_id: str,
        state: dict[str, Any],
    ) -> None:
        self._node_start_times.setdefault(node, []).append(
            time.perf_counter()
        )
        self._dispatch("on_step_start", {
            "type": "step_start",
            "run_id": self.run_id,
            "timestamp": self._timestamp(),
            "node": node,
            "task_id": task_id,
            "state": state,
        })

    def run_end(self, result: dict[str, Any]) -> None:
        self._dispatch("on_run_end", {
            "type": "run_end",
            "run_id": self.run_id,
            "timestamp": self._timestamp(),
            "result": result,
        })

    def run_error(self, error: Exception) -> None:
        self._dispatch("on_run_error", {
            "type": "run_error",
            "run_id": self.run_id,
            "timestamp": self._timestamp(),
            "error_type": type(error).__name__,
            "error": str(error),
        })


class ConsoleTraceHook:
    """Render concise Agent lifecycle events in gray console text."""

    def __init__(self, stream: TextIO | None = None):
        self.stream = stream or sys.stdout

    def _write(self, message: str) -> None:
        print(
            f"{GRAY}[agent] {message}{RESET}",
            file=self.stream,
            flush=True,
        )

    @staticmethod
    def _short(value: Any, limit: int = 100) -> str:
        text = " ".join(str(value or "").split())
        return text if len(text) <= limit else text[:limit - 3] + "..."

    def _summarize(self, event: dict[str, Any]) -> str:
        node = event["node"]
        update = event["update"]
        elapsed = event["elapsed_seconds"]
        prefix = f"{node} ({elapsed:.2f}s)"

        if node == "skill_router":
            skill = update.get("pending_skill") or "none"
            return f"{prefix} skill={skill} reason={update.get('skill_reason')}"

        if node == "skill_loader":
            return f"{prefix} loaded={update.get('active_skill')}"

        if node == "agent":
            if update.get("last_llm_error"):
                return (
                    f"{prefix} llm_error={self._short(update['last_llm_error'])} "
                    f"retry={update.get('llm_retry_pending')}"
                )
            messages = update.get("messages", [])
            message = messages[-1] if messages else None
            if isinstance(message, AIMessage) and message.tool_calls:
                calls = ", ".join(
                    f"{call['name']}({self._short(call.get('args'))})"
                    for call in message.tool_calls
                )
                return f"{prefix} round={update.get('search_round')} tools={calls}"
            content = message.content if isinstance(message, AIMessage) else ""
            return (
                f"{prefix} round={update.get('search_round')} "
                f"candidate={self._short(content)}"
            )

        if node in {"tools", "reject_tools"}:
            messages = update.get("messages", [])
            results = [
                f"{message.name}:{message.status}"
                for message in messages
                if isinstance(message, ToolMessage)
            ]
            return f"{prefix} results={', '.join(results) or 'none'}"

        if node == "record_tools":
            return (
                f"{prefix} total_calls={update.get('tool_calls')} "
                f"evidence={len(update.get('evidence', []))}"
            )

        if node == "evaluate_search_progress":
            progress = update.get("search_progress", {})
            return (
                f"{prefix} reason={progress.get('reason')} "
                f"new_evidence={progress.get('new_evidence_count')} "
                f"new_docs={progress.get('new_document_count')} "
                f"overlap={progress.get('document_overlap')} "
                f"stalled={progress.get('stalled_rounds')} "
                f"sufficient={progress.get('evidence_sufficient')}"
            )

        if node == "validate_answer":
            validation = update.get("answer_validation", {})
            return (
                f"{prefix} valid={validation.get('valid')} "
                f"reason={validation.get('reason')} "
                f"missing={validation.get('missing_terms')} "
                f"retry={update.get('termination_reason') == 'retrying_answer_validation'}"
            )

        return prefix

    def on_run_start(self, event: dict[str, Any]) -> None:
        metadata = event.get("metadata", {})
        self._write(
            f"start run={event['run_id'][:8]} "
            f"model={metadata.get('model', 'unknown')} "
            f"thinking={metadata.get('thinking', 'unknown')}"
        )

    def on_step_start(self, event: dict[str, Any]) -> None:
        self._write(f">> {event['node']}")

    def on_step(self, event: dict[str, Any]) -> None:
        self._write(self._summarize(event))

    def on_run_end(self, event: dict[str, Any]) -> None:
        result = event["result"]
        self._write(
            "end "
            f"reason={result.get('termination_reason')} "
            f"rounds={result.get('search_round')} "
            f"tools={result.get('tool_calls')}"
        )

    def on_run_error(self, event: dict[str, Any]) -> None:
        self._write(
            f"error {event['error_type']}: {self._short(event['error'])}"
        )
