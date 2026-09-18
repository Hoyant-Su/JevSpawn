import argparse
import json
import re
from pathlib import Path


PUBLIC_PAYLOAD_FIELDS = {
    "run_id", "model", "mode", "job", "rank", "world_size", "batch_size",
    "controller_mode", "agent_definition", "gpu_id", "gpu_ids", "tasks",
    "duration_seconds", "input_tokens", "output_tokens", "stage", "choice",
    "options", "probabilities", "probability_status", "output", "truncated",
    "solution", "agent_count", "dataset", "status",
    "spawn_policy", "field_count", "field_mode", "readout_mode", "logical_input_tokens",
    "computed_input_tokens", "padded_input_tokens", "timings", "option_logits",
    "input_token_semantics",
}


def main():
    parser = argparse.ArgumentParser(description="Package recorded execution events for the replay UI.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--architecture", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--public", action="store_true", help="Exclude input prompts and non-public payload fields.")
    parser.add_argument("--standalone", type=Path, help="Also write one offline HTML containing the replay.")
    args = parser.parse_args()
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    events.sort(key=lambda event: event["timestamp"])
    metadata = next(event["payload"] for event in events if event["type"] == "run_started")
    summary = json.loads(args.summary.read_text())
    if args.architecture:
        summary["architecture"] = json.loads(args.architecture.read_text())
    if args.comparison:
        summary["policy_comparison"] = json.loads(args.comparison.read_text())
    if args.profile:
        summary["profile"] = json.loads(args.profile.read_text())
        assert summary["profile"]["run_id"] == summary["run_id"], "The timing profile must match the replay run."
    if args.evaluation:
        summary["evaluation"] = json.loads(args.evaluation.read_text())
        summary["evaluation_status"] = "Evaluated; see evaluation protocol and split-specific results."
    if args.public:
        metadata = {key: value for key, value in metadata.items() if key in PUBLIC_PAYLOAD_FIELDS}
        metadata["export_scope"] = "Recorded task IDs, generated outputs, decisions and timings. Input prompts and reference solutions are not included."
        events = [{**event, "payload": {key: value for key, value in event.get("payload", {}).items() if key in PUBLIC_PAYLOAD_FIELDS}} for event in events]
    replay = {"metadata": metadata, "events": events, "summary": summary}
    encoded = json.dumps(replay, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
    if args.standalone:
        web = Path(__file__).resolve().parents[1] / "web"
        html = (web / "index.html").read_text()
        css = re.sub(r"^@import[^\n]+\n", "", (web / "style.css").read_text(), flags=re.MULTILINE)
        html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>{css}</style>")
        html = html.replace('  <script src="app.js" defer></script>', "")
        embedded = encoded.replace("<", "\\u003c")
        script = (web / "app.js").read_text()
        html = html.replace("</body>", f'<script id="embedded-replay" type="application/json">{embedded}</script>\n<script>{script}</script>\n</body>')
        args.standalone.parent.mkdir(parents=True, exist_ok=True)
        args.standalone.write_text(html)
    print(json.dumps({"output": str(args.output), "event_count": len(events)}))


if __name__ == "__main__":
    main()
