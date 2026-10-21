from io import StringIO

from langchain_core.messages import AIMessage

import agent.graph as graph_module
from agent.hooks import ConsoleTraceHook, GRAY, HookManager, RESET


class RecordingHook:
    def __init__(self):
        self.events = []

    def on_run_start(self, event):
        self.events.append(event)

    def on_step_start(self, event):
        self.events.append(event)

    def on_step(self, event):
        self.events.append(event)

    def on_run_end(self, event):
        self.events.append(event)

    def on_run_error(self, event):
        self.events.append(event)


class FailingHook:
    def on_run_start(self, _event):
        raise RuntimeError("hook failure")

    def on_step_start(self, _event):
        raise RuntimeError("hook failure")

    def on_step(self, _event):
        raise RuntimeError("hook failure")

    def on_run_end(self, _event):
        raise RuntimeError("hook failure")

    def on_run_error(self, _event):
        raise RuntimeError("hook failure")


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        return next(self.responses)


def test_hook_manager_emits_structured_lifecycle_events():
    hook = RecordingHook()
    manager = HookManager(hook)

    manager.run_start(
        "question",
        metadata={"model": "test-model", "thinking": False},
    )
    manager.step_start("agent", "task-1", {"search_round": 0})
    manager.step("agent", {"search_round": 1}, {"final_answer": ""})
    manager.run_end({"final_answer": "answer"})

    assert [event["type"] for event in hook.events] == [
        "run_start",
        "step_start",
        "step",
        "run_end",
    ]
    assert hook.events[2]["node"] == "agent"
    assert hook.events[2]["update"] == {"search_round": 1}
    assert hook.events[2]["state"] == {"final_answer": ""}
    assert hook.events[0]["run_id"] == hook.events[2]["run_id"]
    assert hook.events[0]["metadata"]["thinking"] is False


def test_hook_failure_does_not_break_agent_observation_dispatch():
    recording = RecordingHook()
    manager = HookManager([FailingHook(), recording])

    manager.run_start("question")
    manager.run_end({"final_answer": "answer"})

    assert [event["type"] for event in recording.events] == [
        "run_start",
        "run_end",
    ]


def test_console_hook_uses_gray_text_and_resets_color():
    stream = StringIO()
    hook = ConsoleTraceHook(stream=stream)
    manager = HookManager(hook)

    manager.run_start("question")
    manager.step(
        "evaluate_search_progress",
        {
            "search_progress": {
                "reason": "no_new_evidence",
                "new_evidence_count": 0,
                "new_document_count": 0,
                "document_overlap": 1.0,
                "stalled_rounds": 1,
                "evidence_sufficient": False,
            }
        },
        {},
    )

    output = stream.getvalue()
    assert GRAY in output
    assert RESET in output
    assert "evaluate_search_progress" in output
    assert "stalled=1" in output


def test_run_agent_streams_every_completed_graph_node(monkeypatch):
    model = SequenceModel([
        AIMessage(content=[]),
        AIMessage(content=""),
    ])
    hook = RecordingHook()
    monkeypatch.setattr(graph_module, "get_llm", lambda: model)

    result = graph_module.run_agent(
        "Give me an unsupported fact",
        hooks=hook,
    )

    event_types = [event["type"] for event in hook.events]
    step_nodes = [
        event["node"]
        for event in hook.events
        if event["type"] == "step"
    ]
    assert event_types[0] == "run_start"
    assert event_types[-1] == "run_end"
    assert step_nodes == [
        "skill_router",
        "agent",
        "agent",
        "validate_answer",
    ]
    assert result["final_answer"] == "Unknown"
