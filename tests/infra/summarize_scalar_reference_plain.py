import argparse
from collections import Counter
import json
from pathlib import Path

import yaml


def main(config):
    source = json.loads(Path(config['paired_inputs']).read_text())
    result = json.loads(Path(config['result']).read_text())
    shared = yaml.safe_load(Path(config['shared_config']).read_text())
    reports = {report['variant']: report for report in result['reports']}
    outputs = {name: {value['id']: value for record in report['records']
                     for group in record['result']['groups'] for value in group}
               for name, report in reports.items()}
    rank_equality = {}
    for name, report in reports.items():
        reference = [[value for group in record['result']['groups'] for value in group]
                     for record in report['records']]
        ranks = []
        for rank in range(shared['runtime']['world_size']):
            path = Path(config['output']) / config['rank_file'].format(
                rank=rank, phase=report['phase'], variant=name)
            other = json.loads(path.read_text())
            current = [[value for group in record['result']['groups'] for value in group]
                       for record in other['records']]
            ranks.append(current == reference)
        rank_equality[name] = ranks
        assert all(ranks)
        assert len(outputs[name]) == config['expected_requests']
    rows = []
    for pair in source:
        position = pair['semantic_input']['input']['path']
        expected = pair['semantic_input']['candidates'][position[-1] + config['offline_domain_option_offset']]
        assert 'domain' in expected['value']
        observed = {name: outputs[name][pair['node']] for name in reports}
        rows.append({'node': pair['node'], 'path': position,
                     'expected_domain': expected,
                     'saved_choice': pair['original_result']['choice'],
                     'packed_choice_reproduced': observed['packed']['choice'] == pair['original_result']['choice'],
                     'packed_saved_maximum_absolute_logit_difference': max(abs(a - b) for a, b in zip(
                         observed['packed']['option_logits'], pair['original_result']['option_logits'], strict=True)),
                     'correct_domain': {name: value['choice'] == expected['id'] for name, value in observed.items()},
                     'results': observed})
    report = {'config': config, 'requests': len(rows), 'all_rank_result_equality': rank_equality,
              'original_choice_reproduction': sum(row['packed_choice_reproduced'] for row in rows),
              'original_logit_maximum_absolute_difference': max(row['packed_saved_maximum_absolute_logit_difference'] for row in rows),
              'correct_semantic_domains': {name: sum(row['correct_domain'][name] for row in rows) for name in reports},
              'choice_counts': {name: dict(Counter(row['results'][name]['choice'] for row in rows)) for name in reports},
              'paired_choice_agreement': sum(row['results']['packed']['choice'] == row['results']['plain']['choice'] for row in rows),
              'rows': rows,
              'scope': 'Finite semantic-domain selection only. Correct domains derive offline from original puzzle attribute order; inference received no evaluation labels. Not a full-task quality or speed comparison.'}
    Path(config['summary']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ['config', 'rows']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
