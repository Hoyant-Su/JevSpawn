import argparse
from collections import OrderedDict
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from baselines.common.context_window import truncate_prompt
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.readout_labels import native_labels
from jev_spawn.schema import controller_prefix
from tests.infra.history_prefix import HistoryPrefixTail


def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(shared.model.path, local_files_only=True))
    _, label_ids = native_labels(backend.tokenizer, json.loads(Path(settings['labels']).read_text()))
    backend._render = lambda prompts, system: backend.tokenizer.apply_chat_template(
        [[{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
         for prompt in prompts], **settings['chat_template'])
    tail = SimpleNamespace(backend=backend)
    boundaries = lru_cache(maxsize=shared.runtime.root_batch_size)(
        lambda system, context, history: HistoryPrefixTail._history_tokens(tail, system, context, history))
    prepared = json.loads(Path(settings['prepared']).read_text())
    reports = []
    for workload in prepared['workloads']:
        policy = json.loads(Path(workload['protocol']).read_text())['inference']['settings']['input_window']
        notice = load_prompt(policy['prompt'])
        cache = OrderedDict()
        hits = advances = reused = total = shortened = 0
        for batch in workload['batches']:
            owners = {}
            for packet in batch['requests']:
                rendered = backend.tokenizer.apply_chat_template(packet['messages'], **settings['chat_template'])
                tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
                rendered, tokens, _ = truncate_prompt(backend.tokenizer, rendered, tokens,
                    shared.model.max_input_tokens, policy, notice)
                root = backend.tokenizer.apply_chat_template([packet['messages'][0], {'role': 'user',
                    'content': controller_prefix(packet['field']['context'], '')}], **settings['chat_template'])
                root_tokens = backend.tokenizer(root, add_special_tokens=False)['input_ids']
                assert tokens == packet['tokens'] and rendered == packet['rendered']
                assert tokens[:common_prefix([tokens, root_tokens])] == packet['root_tokens']
                assert label_ids[:len(packet['field']['options'])] == packet['candidate_token_ids']
                owners.setdefault((packet['task_id'], packet['field']['context']), packet)
            for key, packet in owners.items():
                desired = boundaries(packet['messages'][0]['content'], key[1], packet['field']['history'])
                length = max(len(packet['root_tokens']), common_prefix([packet['tokens'][:-1], desired]))
                tokens = tuple(packet['tokens'][:length])
                shortened += length < len(desired)
                sources = [tuple(packet['root_tokens']), *cache.get(key, ())]
                old = max((value for value in sources if tokens[:len(value)] == value), key=len)
                hits += len(old) > len(packet['root_tokens'])
                advances += len(old) > len(packet['root_tokens']) and len(tokens) > len(old)
                reused += len(old) - len(packet['root_tokens'])
                total += len(packet['tokens']) - len(packet['root_tokens'])
                cache[key] = [tokens]
                cache.move_to_end(key)
                if len(cache) > shared.runtime.root_batch_size:
                    cache.popitem(last=False)
        reports.append({'track': workload['track'], 'history_hits': hits,
            'cross_history_extensions': advances, 'reusable_history_tokens': reused,
            'dynamic_tokens_before_reuse': total, 'shortened_history_prefixes': shortened})
    output = Path(settings['output']) / settings['token_boundary_file']
    output.write_text(json.dumps({'scope': 'CPU exact-token ancestry accounting, not inference speed or quality.',
                                 'rows': reports}, indent=2) + '\n')
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
