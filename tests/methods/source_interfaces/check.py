from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from jev_spawn.schema import CONTROLLER
from methods.evidence_flow.environment import EvidenceEnvironment, read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.source_interfaces.inputs import input_size, partition
from methods.source_interfaces.schema import answer_schema
from methods.structured_flow.grammar import SchemaDecoder


stage = read('configs/methods/source_interfaces/stage.json')
fixed, prompts = stage['fixed'], read(stage['prompts'])
tokenizer = AutoTokenizer.from_pretrained(fixed['model_path'])
backend = SimpleNamespace(tokenizer=tokenizer)
backend._render = lambda users, system: tokenizer.apply_chat_template(
    [[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}] for user in users],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
CONTROLLER['option_template'] = prompts['option_template']
configuration = read(Path(fixed['model_path']) / 'config.json')['text_config']
decoder = SchemaDecoder(tokenizer, [configuration['eos_token_id']], configuration['vocab_size'])
for contract in [read(stage['plan_schema']), answer_schema(0), answer_schema(fixed['max_fields_per_round'])]:
    decoder.factory(contract, kind='answer')
environment = EvidenceEnvironment(stage['data']['corpus'], stage['data']['units'], **stage['environment'])
results = []
for row in read_jsonl(stage['data']['tasks']):
    episode = environment.episode(row['task_id'])
    ids = [hit['id'] for hit in episode.search(row['query'], fixed['field_search_count'])]
    units = episode.read(ids)
    groups = partition(backend, units, [('input_check', row['query'])], fixed, prompts)
    represented = [option['id'] for group in groups for option in group[0]['options'][:-1]]
    assert represented == ids and len(set(represented)) == len(ids)
    lengths = [input_size(backend, group, prompts) for group in groups]
    assert max(lengths) <= fixed['context_tokens']
    results.append({'task_id': row['task_id'], 'source_units': len(ids), 'groups': len(groups),
                    'maximum_input_tokens': max(lengths)})
write(Path('results/methods/source_interfaces/cpu_preflight.json'), {'tasks': results, 'grammar_count': 3,
    'scope': 'CPU input partition and grammar validation using original questions as field requests. No generated plan, predictions or task metrics are substituted.'})
print(results)
