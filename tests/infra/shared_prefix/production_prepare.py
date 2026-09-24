import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from baselines.common.context_window import truncate_prompt
from baselines.common.runtime_contract import RuntimeContract
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.readout_labels import native_labels
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prefix, controller_prompts


def messages_for(field, service):
    user, = controller_prompts([field['state']], field['question'], field['options'],
        service.backend.answer_labels, CONTROLLER['output_instruction'],
        contexts=[field['context']], histories=[field['history']])
    return service.contract_messages([
        {'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}])


def decisions(value, settings):
    if isinstance(value, dict):
        for request_key, decision_key in settings['scalar_pairs']:
            if request_key in value and decision_key in value:
                yield value[request_key], value[decision_key]
        if 'requests' in value and 'decisions' in value:
            yield from zip(value['requests'], value['decisions'], strict=True)
        if value.get('kind') == settings['terminal_kind']:
            yield from zip(value['requests'], value['outputs'], strict=True)
        for child in value.values():
            yield from decisions(child, settings)
    elif isinstance(value, list):
        for child in value:
            yield from decisions(child, settings)


def key(task_id, decision):
    return task_id, decision['input_tokens'], tuple(decision['option_logits'])


def prepare(settings):
    shared = SharedConfig.load(settings['shared_config'])
    tokenizer = AutoTokenizer.from_pretrained(shared.model.path, local_files_only=True)
    labels, label_ids = native_labels(tokenizer, json.loads(Path(settings['labels']).read_text()))
    service = RuntimeContract()
    service.shared, service.execution_metadata = shared, {}
    service.backend = SimpleNamespace(answer_labels=labels)
    workloads, coverage = [], []
    for source in settings['sources']:
        run = Path(source['run'])
        protocol = json.loads((run / 'protocol.json').read_text())
        CONTROLLER['option_template'] = protocol['prompts']['option_template']
        input_policy = protocol['inference']['settings']['input_window']
        input_notice = load_prompt(input_policy['prompt'])
        service.configure_runtime_contract(protocol['method']['settings'])
        indexed = defaultdict(list)
        for path in sorted(run.glob(settings['task_pattern'])):
            task = json.loads(path.read_text())
            for field, decision in decisions(task['trace'], settings):
                indexed[key(task['task_id'], decision)].append((field, decision))
        logs = json.loads(Path(source['batches']).read_text())
        selected = [batch for batch in logs if batch.get('operation') == settings['operation']][:settings['batch_count']]
        cache = PrefixCache(shared.runtime.root_batch_size)
        batches, accounting = [], []
        for batch in selected:
            packets = []
            for task_id, decision in zip(batch['task_ids'], batch['structured']['groups'][0], strict=True):
                candidates = [field for field, recorded in indexed[key(task_id, decision)]
                    if recorded['submitted_monotonic'] <= batch['started_monotonic']
                    and recorded['delivered_monotonic'] >= batch['finished_monotonic']]
                assert candidates and all(field == candidates[0] for field in candidates), (
                    source['track'], key(task_id, decision)[:3], len(candidates))
                field = {**settings['request_fields'], **candidates[0]}
                messages = messages_for(field, service)
                rendered = tokenizer.apply_chat_template(messages, **settings['chat_template'])
                tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
                rendered, tokens, _ = truncate_prompt(tokenizer, rendered, tokens,
                    shared.model.max_input_tokens, input_policy, input_notice)
                assert len(tokens) == decision['input_tokens'], (
                    source['track'], task_id, field['id'], len(tokens), decision['input_tokens'])
                root = tokenizer.apply_chat_template([messages[0], {'role': 'user',
                    'content': controller_prefix(field['context'], '')}], **settings['chat_template'])
                root_length = common_prefix([tokens, tokenizer(root, add_special_tokens=False)['input_ids']])
                packets.append({'field': field, 'messages': messages, 'rendered': rendered, 'tokens': tokens,
                    'root_tokens': tokens[:root_length], 'task_id': task_id,
                    'candidate_token_ids': label_ids[:len(field['options'])]})
            groups = defaultdict(list)
            for packet in packets:
                groups[(packet['task_id'], packet['field']['context'])].append(packet)
            prefixes, bases = [], []
            for group in groups.values():
                sequences = [packet['tokens'] for packet in group]
                length = min(common_prefix(sequences), min(map(len, sequences)) - 1)
                prefixes.append(sequences[0][:length])
                bases.append(group[0]['root_tokens'])
            _, hit = cache.get([*bases, *prefixes], lambda: None)
            count = sum(len(prefix) - len(base) for prefix, base in zip(prefixes, bases, strict=True))
            accounting.append({'batch_size': len(packets), 'cohort_hit': hit,
                'shared_extension_tokens': count, 'avoidable_tokens': hit * count})
            batches.append({'requests': packets})
        workloads.append({'track': source['track'], 'source': source['batches'],
            'protocol': str(run / 'protocol.json'), 'batches': batches})
        coverage.append({'track': source['track'], 'sample_count': len({packet['task_id']
            for batch in batches for packet in batch['requests']}), 'batches': accounting})
        print(json.dumps({'track': source['track'], 'completed_batches': len(batches),
            'hit_batches': sum(row['cohort_hit'] for row in accounting)}), flush=True)
    output = Path(settings['prepared'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'workloads': workloads, 'measurement_scope': settings['scope']}) + '\n')
    Path(settings['coverage']).write_text(json.dumps({'scope': settings['scope'], 'workloads': coverage}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    prepare(json.loads(parser.parse_args().config.read_text()))
