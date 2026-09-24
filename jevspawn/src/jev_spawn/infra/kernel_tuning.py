import inspect
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import triton
from triton.runtime.autotuner import Autotuner, Heuristics


def tuners(settings):
    found = {}
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith(tuple(settings['module_prefixes'])):
            continue
        for value in vars(module).values():
            while isinstance(value, Heuristics):
                value = value.fn
            if isinstance(value, Autotuner):
                key = value.base_fn.__module__ + '.' + value.base_fn.__qualname__
                if key in found and found[key] is not value:
                    raise ValueError(f'Ambiguous native autotuner identity: {key}')
                found[key] = value
    return found


def identity(backend):
    properties = torch.cuda.get_device_properties(backend.device)
    return {'triton': triton.__version__, 'cuda': torch.version.cuda,
            'gpu': properties.name, 'capability': [properties.major, properties.minor],
            'rank': dist.get_rank(), 'world_size': dist.get_world_size(),
            'parameters': [[name, list(value.shape), str(value.dtype)]
                           for name, value in backend.model.named_parameters()]}


def signature(tuner):
    return {'source': inspect.getsource(tuner.base_fn), 'keys': tuner.keys,
            'arguments': tuner.arg_names,
            'configurations': [{'arguments': config.all_kwargs(),
                                'pre_hook': None if config.pre_hook is None else
                                config.pre_hook.__module__ + '.' + config.pre_hook.__qualname__}
                               for config in tuner.configs]}


def save_tuning(path, backend, settings):
    records = {}
    for name, tuner in tuners(settings).items():
        if not tuner.cache:
            continue
        entries = []
        for key, chosen in tuner.cache.items():
            matches = [index for index, config in enumerate(tuner.configs)
                       if config.all_kwargs() == chosen.all_kwargs() and config.pre_hook is chosen.pre_hook]
            if len(matches) != settings['singleton_count']:
                raise ValueError(f'Tuned configuration is not uniquely in the native catalog: {name}')
            entries.append({'key': list(key), 'configuration_index': matches[settings['first_index']]})
        records[name] = {'signature': signature(tuner), 'entries': entries}
    payload = {'identity': identity(backend), 'tuners': records}
    Path(path).write_text(json.dumps(payload, **settings['serialization']) + '\n')
    return {'tuners': len(records), 'entries': sum(len(record['entries']) for record in records.values())}


def load_tuning(path, backend, settings):
    payload = json.loads(Path(path).read_text())
    if payload['identity'] != identity(backend):
        raise ValueError('Native tuning cache model, shard, hardware or version identity differs.')
    current = tuners(settings)
    for name, record in payload['tuners'].items():
        tuner = current[name]
        if record['signature'] != signature(tuner):
            raise ValueError(f'Native autotuner source or configuration catalog differs: {name}')
        for entry in record['entries']:
            key = tuple(entry['key'])
            chosen = tuner.configs[entry['configuration_index']]
            if key in tuner.cache and tuner.cache[key].all_kwargs() != chosen.all_kwargs():
                raise ValueError(f'Existing native cache conflicts with persisted selection: {name}')
            tuner.cache[key] = chosen
    return {'tuners': len(payload['tuners']),
            'entries': sum(len(record['entries']) for record in payload['tuners'].values())}
