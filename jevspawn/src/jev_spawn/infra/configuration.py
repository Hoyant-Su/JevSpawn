from importlib import import_module
import json

from jev_spawn.infra.prompts import ROOT, resolve_prompts


RESOURCES = json.loads((ROOT / 'configs/jevspawn/resources.json').read_text())


def load_resource(name):
    return resolve_prompts(json.loads((ROOT / RESOURCES[name]).read_text()))


def resolve_symbol(path):
    module, name = path.rsplit('.', maxsplit=1)
    return getattr(import_module(module), name)


CORE = load_resource('core')
LANGUAGE = load_resource('language')
EXECUTION_POLICY = load_resource('execution_policy')
