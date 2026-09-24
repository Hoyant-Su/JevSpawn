import json

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt


TEMPLATES = load_prompt('configs/baselines/common/templates/interface.json')
ADAPTER_SETTINGS = json.loads((ROOT / 'configs/baselines/common/runtime_adapters.json').read_text())
ANSWER_PROPERTIES = json.loads((ROOT / 'configs/baselines/common/schema/answer_properties.json').read_text())


def snapshot():
    return {'templates': TEMPLATES, 'adapter_settings': ADAPTER_SETTINGS,
            'answer_properties': ANSWER_PROPERTIES,
            'core': json.loads((ROOT / 'configs/jevspawn/core.json').read_text()),
            'reproducibility': json.loads((ROOT / 'configs/reproducibility.json').read_text())}
