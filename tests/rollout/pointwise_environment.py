import argparse
from functools import partial
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_run import initialize_parallel
from baselines.common.runtime import InferenceRuntime
from jev_spawn.algo.structured import padded
from jev_spawn.infra.configuration import resolve_symbol
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.semantic_domain import compile_domain
from jev_spawn.rollout.query import declare_domain
from jev_spawn.runtime.observation_memory import ObservationMemory


def read(path):
    return json.loads(Path(path).read_text())


def run(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, read(config['parallel_settings']))
    method, settings = read(config['method']), read(config['rollout'])
    prompts = load_prompt(settings['prompts'])
    domain = compile_domain(backend.tokenizer, config['values'])
    weights = backend.selected_output_weights(domain.branches).float()
    positive = domain.values.index(config['positive_value'])
    runtime = InferenceRuntime(config['shared_config'], partial(StableGraphFiniteService,
        settings=method['settings'], prompts=load_prompt(method['prompts'])), backend=backend)

    @torch.inference_mode()
    def score(payload):
        sequences = []
        for messages in payload['messages']:
            rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
            tokens = backend.tokenizer(rendered, add_special_tokens=False)['input_ids']
            for value, tail in zip(domain.values, domain.tails, strict=True):
                assert backend.tokenizer(rendered + value, add_special_tokens=False)['input_ids'] == tokens + list(domain.prefix) + list(tail)
            sequences.append(tokens + list(domain.prefix))
        assert len(sequences) <= shared.runtime.batch_size
        assert max(map(len, sequences)) <= shared.model.max_input_tokens
        ids, mask = padded(sequences, backend.tokenizer.pad_token_id, backend.device, config['padding'])
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        output = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1)-1).clamp_min(0), use_cache=False)
        logits = F.linear(output.last_hidden_state[:, -1].float(), weights)
        scores = logits.log_softmax(-1)[:, positive]
        chosen = torch.stack([group.argmax() for group in scores.split(payload['counts'])]).tolist()
        torch.cuda.synchronize(backend.device)
        return {'chosen': chosen, 'scores': scores.tolist(), 'logits': logits.tolist(),
                'seconds': time.perf_counter()-started, 'input_tokens': list(map(len, sequences))}

    commands.register(config['command'], score)
    if commands.is_leader:
        output = Path(config['output'])
        output.mkdir(parents=True, exist_ok=True)
        factory, contract = read(settings['environment']['factory']), settings['environment']
        tasks = [json.loads(line) for line in Path(contract['tasks']).read_text().splitlines()]
        assert len(tasks) == config['task_count']
        environments = [resolve_symbol(factory['class'])(task, {}, output,
            deadline=partial(runtime.deadlines.remaining, task['task_id']), **factory['parameters']) for task in tasks]
        records = []
        memories = []
        for task, environment in zip(tasks, environments, strict=True):
            task_id = task['task_id']
            runtime.deadlines.start(task_id)
            query = environment.context(False, environment.configuration['serialization'])
            trace = {'task_id': task_id, 'query': query, 'history': []}
            trace['actions'] = declare_domain(query, complete=runtime.complete(task_id), settings=settings,
                prompts=prompts, budget={'max_new_tokens': shared.generation.max_new_tokens,
                'temperature': shared.generation.temperature}, trace=trace)
            records.append(trace)
            memories.append(ObservationMemory({'observation': environment.observation},
                trace['actions'], config['serialization']))
        for turn in range(shared.runtime.max_turns):
            active = [(record, environment, memory) for record, environment, memory in
                      zip(records, environments, memories, strict=True) if not environment.done]
            if not active:
                break
            messages = []
            for record, environment, memory in active:
                runtime.deadlines.remaining(record['task_id'])
                state = prompts['observation_memory'].format(memory=json.dumps(memory.view(), **config['serialization']))
                context = prompts['observed_task'].format(context=record['query'], state=state)
                messages.extend([{'role': 'system', 'content': prompts['direct']}, {'role': 'user',
                    'content': prompts['atomic_action_preference'].format(context=context, action=action)}]
                    for action in record['actions'])
            result = commands.call(config['command'], {'messages': messages, 'counts': [len(record['actions']) for record, _, _ in active]})
            for (record, environment, memory), choice in zip(active, result['chosen'], strict=True):
                action = record['actions'][choice]
                feedback, done = environment.execute(contract['tool'], {contract['argument']: action})
                record['history'].append({'action': action, 'feedback': json.loads(feedback)})
                memory.record(action, json.loads(feedback))
                record['memory'] = memory.view()
                record.update(terminal=done, answer=environment.answer)
            (output / config['step_file'].format(turn=turn)).write_text(json.dumps({'result': result, 'records': records}, indent=2)+'\n')
        (output / 'completion.json').write_text(json.dumps({'config': config, 'startup': startup, 'records': records}, indent=2)+'\n')
        commands.finish()
    else:
        commands.serve()
    runtime.close()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(read(parser.parse_args().config))
