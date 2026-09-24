import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
from tokenizers import Tokenizer

from methods.structured_flow.grammar import SchemaDecoder, decoder_schema

from project_paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[5]
    raw = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    configuration = json.loads((args.tokenizer / "config.json").read_text())["text_config"]
    token_configuration = json.loads((args.tokenizer / "tokenizer_config.json").read_text())
    tokenizer = SimpleNamespace(backend_tokenizer=raw, pad_token_id=raw.token_to_id(token_configuration["pad_token"]))
    eos_ids = [configuration["eos_token_id"]]
    started = time.perf_counter()
    decoder = SchemaDecoder(tokenizer, eos_ids, configuration["vocab_size"])
    initialized = time.perf_counter() - started
    source = json.loads((root / "research/jevspawn-paper/jevspawn/src/jev_spawn/schema/flow.json").read_text())
    assert decoder_schema(source, "flow") == json.loads((ROOT / "configs/methods/structured_flow/schema/flow_decoder.json").read_text())
    leaf = {"nodes": [{"id": "answer", "kind": "collect", "depends_on": [], "input": {"$item": True}}], "output": "answer"}
    graph = {"nodes": [
        {"id": "fan", "kind": "map", "depends_on": [], "input": {"$task": ["input", "items"]}, "items_path": [], "template": leaf},
        {"id": "count", "kind": "reduce", "depends_on": ["fan"], "input": {"$result": "fan"}, "operator": "count_equal", "field_path": [], "equals": True},
    ], "output": "count"}
    option_graph = {"nodes": [{"id": "check", "kind": "decide", "depends_on": [], "input": None,
        "question": "Select one.", "options": [{"id": str(i), "description": str(i)} for i in range(26)],
        "branches": {"0": leaf}}], "output": "check"}
    tasks = [json.loads(line) for line in (ROOT / 'data/methods/structured_flow/development_tasks.jsonl').read_text().splitlines()]
    fixtures = [
        (source, "flow", graph), (source, "flow", option_graph),
        (tasks[0]["answer_contract"], "answer", {"code": "def f(x):\n    return x + 1\n"}),
        (tasks[2]["answer_contract"], "answer", {key: spec["enum"][0] for key, spec in tasks[2]["answer_contract"]["properties"].items()}),
        (tasks[4]["answer_contract"], "answer", {key: spec["enum"][0] for key, spec in tasks[4]["answer_contract"]["properties"].items()}),
        (tasks[7]["answer_contract"], "answer", {"rows": [["fixture"] * tasks[7]["answer_contract"]["properties"]["rows"]["items"]["minItems"] for _ in range(tasks[7]["answer_contract"]["properties"]["rows"]["minItems"])]}),
    ]
    prefix = [tokenizer.pad_token_id] * 5
    calls, mask_seconds = 0, 0.0
    for schema, kind, value in fixtures:
        jsonschema.validate(value, schema)
        factory = decoder.factory(schema, kind=kind)
        constraint = factory(tokenizer, eos_ids, len(prefix), 4096)
        tokens = raw.encode(json.dumps(value, separators=(",", ":")), add_special_tokens=False).ids
        for index, token in enumerate(tokens):
            before = time.perf_counter()
            allowed = constraint(0, np.array(prefix + tokens[:index]))
            mask_seconds += time.perf_counter() - before
            calls += 1
            assert token in allowed, (kind, index, raw.decode([token]))
            assert not set(eos_ids).intersection(allowed)
        assert eos_ids[0] in constraint(0, np.array(prefix + tokens))
        assert not set(eos_ids).intersection(constraint(1, np.array(prefix)))
        assert constraint(0, np.array(prefix + tokens + eos_ids + [tokenizer.pad_token_id])) == eos_ids
        assert constraint.rows[0]["matcher"] is not constraint.rows[1]["matcher"]
    for invalid in ['{"$task":true}', '```json\n{}\n```', '{"nodes":[],"output":"x"}']:
        constraint = decoder.factory(source, kind="flow")(tokenizer, eos_ids, 5, 4096)
        tokens = raw.encode(invalid, add_special_tokens=False).ids
        blocked = False
        for index, token in enumerate(tokens):
            if token not in constraint(0, np.array(prefix + tokens[:index])):
                blocked = True
                break
        assert blocked, invalid
    for task in tasks:
        decoder.factory(task["answer_contract"], kind="answer")
    duplicate = {"nodes": [{"id": "a", "kind": "collect", "depends_on": ["x", "x"], "input": None}], "output": "a"}
    assert list(jsonschema.Draft202012Validator(source).iter_errors(duplicate))
    result = {"valid_fixtures": len(fixtures), "invalid_sequences_rejected": 3,
              "answer_contracts_compiled": len(tasks), "per_row_state_and_eos_padding": True,
              "original_uniqueness_validator_retained": True, "model_loaded": False,
              "n_vocab": configuration["vocab_size"], "eos_ids": eos_ids,
              "tokenizer_initialization_seconds": initialized,
              "mask_calls": calls, "mask_total_seconds": mask_seconds,
              "mean_mask_ms": mask_seconds / calls * 1000}
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
