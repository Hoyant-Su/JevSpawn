import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from jev_spawn.infra.prompts import PROMPTS, load_prompt
from tests.methods.observed_fields.candidate import install


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    prompt_key = PROMPTS['references'][settings['replaced_prompt']]
    PROMPTS['templates'][prompt_key] = load_prompt(settings['prompt'])
    install()
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
