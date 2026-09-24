import json
from unittest.mock import patch

from baselines.latentmas.common_service import LatentMASService
from baselines.latentmas.context_window_service import WindowedLatentMASService
from jev_spawn.infra.configuration import ROOT
from jev_spawn.infra.prompts import load_prompt


def test_canonical_input_notice_is_resolved_leaf():
    inference = json.loads((ROOT / 'configs/inference/shared_service.json').read_text())
    policy = inference['settings']['input_window']
    expected = load_prompt(policy['prompt'])
    assert isinstance(expected, str) and expected
    service = object.__new__(WindowedLatentMASService)
    service.execution_metadata = {}
    with patch.object(LatentMASService, '__init__', return_value=None):
        service.__init__(None, None, None, settings={}, prompts={}, input_policy=policy)
    assert service.input_notice == expected
    assert service.execution_metadata['input_policy'] == policy
