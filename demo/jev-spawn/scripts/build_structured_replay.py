import argparse
import json
import statistics
from pathlib import Path


def public_paths(value):
    if isinstance(value, dict):
        return {key: Path(item).name if key in {"model", "model_path"} and isinstance(item, str) and Path(item).is_absolute() else public_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_paths(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description="Package recorded structured-worker events and measured results for an offline replay.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standalone", type=Path)
    args = parser.parse_args()
    events = [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    events.sort(key=lambda event: event["timestamp"])
    summary = json.loads(args.summary.read_text())
    starts = {event["agent_id"]: event["timestamp"] for event in events if event["type"] == "agent_started"}
    latencies = [event["timestamp"] - starts[event["agent_id"]] for event in events if event["type"] == "agent_completed"]
    if latencies:
        summary["median_worker_latency_seconds"] = statistics.median(latencies)
    evaluation = json.loads(args.evaluation.read_text()) if args.evaluation else None
    if evaluation is not None:
        summary["evaluation_status"] = "Evaluated; see the supplied quality artifact and metric conventions."
    comparison = json.loads(args.comparison.read_text()) if args.comparison else None
    replay = public_paths({"events": events, "summary": summary, "evaluation": evaluation, "comparison": comparison})
    encoded = json.dumps(replay, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
    if args.standalone:
        template = Path(__file__).resolve().parents[1] / "web" / "structured.html"
        html = template.read_text()
        embedded = '<script id="structured-replay" type="application/json">' + encoded.replace("<", "\\u003c") + "</script>\n"
        html = html.replace("<script>\n'use strict';", embedded + "<script>\n'use strict';")
        args.standalone.parent.mkdir(parents=True, exist_ok=True)
        args.standalone.write_text(html)
    print(json.dumps({"events": len(events), "run_id": summary["run_id"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
