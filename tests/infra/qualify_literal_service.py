import argparse
import json
from pathlib import Path
from unittest.mock import patch

from baselines.common.jevspawn_service import StructuredService
from baselines.common.literal_state_service import LiteralStateService
from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.prompts import load_prompt


def admitted_fields(service, fields, *, task_id):
    return fields


def main(config):
    original = [json.loads(path.read_text()) for path in sorted(Path(config['source_run']).glob(config['input_glob']))]
    expected = json.loads(Path(config['records']).read_text())
    holder = object.__new__(LiteralStateService)
    holder.literal_prompts = load_prompt(config['prompts'])
    holder.literal_transport = load_resource(config['transport'])
    holder.literal_shared_keys = config['shared_keys']
    holder.serialization = load_resource('finite_control')['serialization']
    matches = []
    with patch.object(StructuredService, 'decide', admitted_fields):
        for old, new in zip(original, expected, strict=True):
            for source, target in zip(old['requests'], new['requests'], strict=True):
                field, = LiteralStateService.decide(holder, [source['field']], task_id=source['task_id'])
                assert field == target['field']
                matches.append([source['task_id'], field['id']])
    report = {'requests': len(matches), 'exact_fields': matches,
              'scope': 'Actual service transformation and task-context wrapping match every previously tokenized complete request. GPU admission and numerical execution are separate.'}
    Path(config['service_result']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'requests': len(matches), 'exact_service_fields': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
