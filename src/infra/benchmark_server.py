import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def percentiles(values):
    return dict(zip(["median", "p95"], np.percentile(values, [50, 95]).tolist())) if values else None


async def complete(session, config, row):
    payload = {
        "model": config["model"], "prompt": row["prompt"],
        "temperature": 0, "top_p": 1, "seed": config["seed"],
        "max_tokens": config["max_tokens"], "n": 1, "echo": False, "add_special_tokens": False,
        "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True,
        "structured_outputs": row["structured_outputs"],
    }
    chunks, texts, finish_reason, usage = [], [], None, None
    started = time.perf_counter()
    async with session.post(config["base_url"] + "/v1/completions", json=payload) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage") is not None:
                usage = event["usage"]
            for choice in event["choices"]:
                texts.append(choice["text"])
                if choice["finish_reason"] is not None:
                    finish_reason = choice["finish_reason"]
                if choice["token_ids"]:
                    chunks.append({"seconds": time.perf_counter() - started,
                                   "tokens": choice["token_ids"]})
    assert chunks and finish_reason is not None and usage is not None
    # Chunk coalescing makes individual token times unobservable; never divide a chunk gap into fake token times.
    intervals = [right["seconds"] - left["seconds"] for left, right in zip(chunks, chunks[1:])
                 if len(left["tokens"]) == len(right["tokens"]) == 1]
    return {
        "id": row["id"], "task_id": row["task_id"], "field_names": row["field_names"],
        "text": "".join(texts), "finish_reason": finish_reason,
        "usage": usage, "elapsed_seconds": time.perf_counter() - started,
        "first_token_or_chunk_seconds": chunks[0]["seconds"],
        "first_chunk_token_count": len(chunks[0]["tokens"]),
        "stream_chunks": chunks, "observable_inter_token_seconds": intervals,
        "observable_inter_token_ms": percentiles([value * 1000 for value in intervals]),
        "inter_token_over_100ms": sum(value >= 0.1 for value in intervals),
    }


async def run(config):
    rows, warmup = read_rows(config["requests"]), read_rows(config["warmup_requests"])
    assert len({row["id"] for row in rows}) == len(rows)
    assert rows and warmup and config["batch_size"] > 0
    assert not set(row["prompt"] for row in warmup) & set(row["prompt"] for row in rows)
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    timeout = aiohttp.ClientTimeout(total=config["timeout_seconds"])
    connector = aiohttp.TCPConnector(limit=config["batch_size"])
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for start in range(0, len(warmup), config["batch_size"]):
            await asyncio.gather(*(complete(session, config, row)
                                   for row in warmup[start:start + config["batch_size"]]))
        for index, start in enumerate(range(0, len(rows), config["batch_size"])):
            batch = rows[start:start + config["batch_size"]]
            output = root / f"batch-{index:05d}.json"
            if output.exists():
                assert [row["id"] for row in json.loads(output.read_text())["rows"]] == [row["id"] for row in batch]
                continue
            started = time.perf_counter()
            results = await asyncio.gather(*(complete(session, config, row) for row in batch))
            record = {"batch_index": index, "batch_size": len(batch),
                      "elapsed_seconds": time.perf_counter() - started, "rows": results}
            temporary = output.with_suffix(".partial")
            temporary.write_text(json.dumps(record) + "\n")
            temporary.replace(output)
            print(json.dumps({key: value for key, value in record.items() if key != "rows"}), flush=True)
    batches = [json.loads((root / f"batch-{index:05d}.json").read_text())
               for index in range((len(rows) + config["batch_size"] - 1) // config["batch_size"])]
    results = [row for batch in batches for row in batch["rows"]]
    intervals = [value for row in results for value in row["observable_inter_token_seconds"]]
    chunks = [chunk for row in results for chunk in row["stream_chunks"]]
    summary = {
        "samples": len(results), "batch_sizes": [batch["batch_size"] for batch in batches],
        "request_unit": config["request_unit"],
        "root_tasks": len({row["task_id"] for row in results}),
        "field_decisions": sum(len(row["field_names"]) for row in results),
        "sum_batch_wall_seconds": sum(batch["elapsed_seconds"] for batch in batches),
        "prompt_tokens": sum(row["usage"]["prompt_tokens"] for row in results),
        "completion_tokens": sum(row["usage"]["completion_tokens"] for row in results),
        "first_token_or_chunk_ms": percentiles([row["first_token_or_chunk_seconds"] * 1000 for row in results]),
        "observable_inter_token_ms": percentiles([value * 1000 for value in intervals]),
        "observable_inter_token_count": len(intervals),
        "total_streamed_token_intervals": sum(
            max(0, sum(len(chunk["tokens"]) for chunk in row["stream_chunks"]) - 1) for row in results),
        "streamed_token_count": sum(len(chunk["tokens"]) for chunk in chunks),
        "coalesced_chunk_count": sum(len(chunk["tokens"]) > 1 for chunk in chunks),
        "inter_token_over_100ms": sum(value >= 0.1 for value in intervals),
        "length_limited_samples": sum(row["finish_reason"] == "length" for row in results),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(json.loads(args.config.read_text())))
