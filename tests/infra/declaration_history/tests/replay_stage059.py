import argparse
from collections import Counter
import json
import time

from jev_spawn.schema.declaration_builder import DeclarationBuilder
from replay_stage055 import ROOT, canonical, replay


class CappedBuilder(DeclarationBuilder):
    def field(self, role, values):
        assert len(self.program['fields']) < self.settings['max_fields'], 'Declaration field capacity exhausted.'
        return super().field(role, values)


def outcome(builder, task, revision, protocol):
    try:
        declaration, calls, compiled = replay(builder, task, revision, protocol)
    except (AssertionError, ValueError, SyntaxError) as error:
        return {'accepted': False, 'error': str(error), 'error_type': type(error).__name__}
    return {'accepted': True, 'declaration': declaration, 'compiled_calls': calls,
            'compiled': compiled, 'field_count': len(declaration['fields'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    rows, counts = [], Counter()
    started = time.perf_counter()
    for run in config['runs']:
        directory = ROOT / run
        protocol = json.loads((directory / 'protocol.json').read_text())
        for path in sorted(directory.glob('task-*.json')):
            task = json.loads(path.read_text())
            for turn in task['trace']['rounds']:
                revision = turn.get('revision', {})
                observation = revision.get('observation', {})
                historical_capacity = observation.get('error') == config['capacity_error']
                if not observation.get('accepted') and not historical_capacity:
                    continue
                capped = outcome(CappedBuilder, task, revision, protocol)
                uncapped = outcome(DeclarationBuilder, task, revision, protocol)
                exact = canonical(capped) == canonical(uncapped)
                assert exact, (str(path), turn['turn'], capped, uncapped)
                counts['paired_exact'] += 1
                if historical_capacity:
                    counts['historical_capacity_rejections'] += 1
                    counts['historical_capacity_now_accepted_by_both'] += capped['accepted']
                else:
                    counts['recorded_accepted'] += 1
                rows.append({'source': str(path.relative_to(ROOT)), 'task_id': task['task_id'],
                    'turn': turn['turn'], 'historical_capacity_rejection': historical_capacity,
                    'current_capped_accepted': capped['accepted'], 'uncapped_accepted': uncapped['accepted'],
                    'paired_exact': exact, 'source_signatures': observation['source_signatures'],
                    'uncapped_result': uncapped})
    report = {'stage': config['stage'], 'config': args.config, 'counts': dict(counts),
        'elapsed_seconds': time.perf_counter() - started,
        'scope': 'Replay actual recorded model outputs through the stage055 compiler with its recorded cap and the otherwise identical stage059 compiler without a declaration-wide field cap. No new model calls or invented outputs.',
        'conclusion': 'The historical capacity rejections already compile under the current capped compiler. This replay establishes unchanged compilation for these recorded outputs; fresh inference must test whether removing the declared cap changes model-proposed command coverage.',
        'limits': 'Compiler acceptance checks signature language and references, not correctness of native punctuation, action arity, argument domains, state applicability or final task success.',
        'rows': rows}
    (ROOT / config['output']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
