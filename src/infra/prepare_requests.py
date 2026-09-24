import argparse
import json
import string
from pathlib import Path

from transformers import AutoTokenizer

from jev_spawn.schema import BENCHMARK, CONTROLLER, controller_prompts, joint_controller_prompts


def requests(task, tokenizer, mode):
    fields = task["fields"]
    if mode == "compact_json":
        prompts = joint_controller_prompts([task["state"]], fields, string.ascii_uppercase,
                                           BENCHMARK["array_output_instruction"])
        menus = [list(string.ascii_uppercase[:len(field["options"])]) for field in fields.values()]
        pattern = r'\["' + r'","'.join("[" + "".join(menu) + "]" for menu in menus) + r'"\]'
        specifications = [(task["task_id"], prompts[0], {"regex": pattern}, list(fields))]
    else:
        specifications = []
        for name, field in fields.items():
            labels = list(string.ascii_uppercase[:len(field["options"])])
            prompt = controller_prompts([task["state"]], field["question"], field["options"], labels,
                                        CONTROLLER["output_instruction"])[0]
            specifications.append((task["task_id"] + ":" + name, prompt, {"choice": labels}, [name]))
    result = []
    for identity, prompt, constraint, names in specifications:
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": CONTROLLER["system"]}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        result.append({"id": identity, "task_id": task["task_id"], "field_names": names,
                       "prompt": rendered, "structured_outputs": constraint})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["compact_json", "single_label"], required=True)
    parser.add_argument("--task-offset", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--max-input-tokens", type=int, required=True)
    args = parser.parse_args()
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines()]
    tasks = tasks[args.task_offset:args.task_offset + args.task_count]
    assert len(tasks) == args.task_count
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = [row for task in tasks for row in requests(task, tokenizer, args.mode)]
    lengths = [len(ids) for ids in tokenizer([row["prompt"] for row in rows], add_special_tokens=False)["input_ids"]]
    assert max(lengths) <= args.max_input_tokens
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"tasks": len(tasks), "requests": len(rows), "max_input_tokens": max(lengths)}))


if __name__ == "__main__":
    main()
