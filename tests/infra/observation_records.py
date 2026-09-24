import argparse
import json
from pathlib import Path

from baselines.common.parallel_run import execute
from jev_spawn.infra.prompts import PROMPTS, load_prompt
from jev_spawn.rollout import branching
from jev_spawn.runtime import query_execution


def literal_history(events, preceding):
    protocol = load_prompt('jevspawn.state')
    return ''.join(protocol['history_record'].format(
        event=json.dumps(event, **protocol['history_serialization'])) for event in events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    if settings['literal_observations']:
        branching.extend_event_history = query_execution.extend_event_history = literal_history
        PROMPTS['templates']['jevspawn.state']['history_format'] = load_prompt(settings['state_prompt'])
    specification = json.loads(Path(settings['specification']).read_text())
    assert specification['shared_config'] == settings['shared_config']
    execute(specification, Path(settings['run_output']),
            json.loads(Path(settings['parallel_settings']).read_text()))
