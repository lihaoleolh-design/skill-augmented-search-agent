import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

file_path = (
    ROOT
    / "output"
    / "trajectories"
    / "20261019_134516_615920.json"
)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Question:", data["question"])
print("Answer:", data["answer"])

print("\nAgent Execution Timeline")
print("=" * 60)

for i, event in enumerate(data["trajectory"], 1):
    event_type = event.get("type")

    print(f"\nStep {i}: {event_type}")

    if event_type == "llm_response":
        print("Round:", event.get("round"))
        print("Tokens:", event.get("total_tokens"))
        print("Tool requests:", [
            call["name"]
            for call in event.get("tool_calls", [])
        ])

    elif event_type == "tool_call":
        print("Tool:", event.get("tool"))
        print("Input:", event.get("input"))

    elif event_type == "tool_result":
        print("Tool:", event.get("tool"))
        output = event.get("output")
        print(
            "Output length:",
            len(json.dumps(output, ensure_ascii=False))
        )