import json

from jev_spawn.infra.prompts import ROOT


ADAPTER_SETTINGS = json.loads((ROOT / 'configs/completion.json').read_text())
