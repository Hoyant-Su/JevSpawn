from functools import partial

from jev_spawn.infra.configuration import resolve_symbol
from jev_spawn.infra.prompts import load_prompt


def service_factory(method, inference):
    mode = method['inference_service']
    constructor = resolve_symbol(inference['services'][mode])
    if mode == 'shared':
        return partial(constructor, settings=inference['settings'],
                       prompts=load_prompt(inference['prompts']))
    assert mode == 'latentmas'
    return partial(constructor, settings=method['settings'],
                   prompts=load_prompt(method['prompts']),
                   input_policy=inference['settings']['input_window'])
