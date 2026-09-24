import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from baselines.common.environment import TaskEnvironment
from baselines.common.tasks import read, rows
from baselines.latentmas.common_service import role_messages
from methods.evidence_flow.environment import EvidenceEnvironment
from jev_spawn.infra.prompts import load_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--shared-config', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--method', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    shared, method, tools = SharedConfig.load(args.shared_config), read(args.method), read(args.environment)
    prompts = load_prompt(method['prompts'])
    tasks = rows(args.tasks)
    tokenizer = AutoTokenizer.from_pretrained(shared.model.path, local_files_only=True)
    evidence = EvidenceEnvironment(**tools['evidence'])
    records = []
    for task in tasks:
        environment = TaskEnvironment(task, tools, args.output.parent / 'tools' / task['task_id'],
                                      deadline=lambda: None, evidence=evidence)
        messages = [{'role': 'system', 'content': prompts['system']},
                    {'role': 'user', 'content': environment.reset()}]
        roles = role_messages(messages, prompts, shared.model.path)
        rendered = [tokenizer.apply_chat_template(role, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) for role in roles]
        encoded = tokenizer(rendered, add_special_tokens=False, truncation=False)['input_ids']
        lengths = list(map(len, encoded))
        required = sum(lengths) + 3 * method['settings']['latent_steps']
        records.append({'task_id': task['task_id'], 'dataset': task['dataset'], 'role_tokens': lengths,
                        'required_context_tokens': required, 'context_fits': required <= shared.model.max_input_tokens,
                        'available_tools': environment.tools, 'labels_loaded': False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope': 'CPU tokenizer and real environment initial-turn preflight; no model inference',
        'model': shared.model.path, 'max_input_tokens': shared.model.max_input_tokens, 'max_new_tokens': shared.generation.max_new_tokens,
        'tasks': len(records), 'fits': sum(row['context_fits'] for row in records), 'records': records}, indent=2) + '\n')
    print(json.dumps({'tasks': len(records), 'fits': sum(row['context_fits'] for row in records),
                      'rejected': [row['dataset'] for row in records if not row['context_fits']]}))


if __name__ == '__main__':
    main()
