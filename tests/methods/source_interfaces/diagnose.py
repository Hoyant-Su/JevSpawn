import argparse
import json
from pathlib import Path
import statistics

from methods.source_interfaces.schema import NONE


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    diagnosis = read(args.stage)
    stage = read(diagnosis['development_stage'])
    evaluation = read(diagnosis['evaluation'])
    labels = [json.loads(line) for line in Path(stage['data']['labels']).read_text().splitlines()]
    assert Path(stage['data']['tasks']).parent.name == 'development'
    assert evaluation['task_count'] == len(labels) == stage['data']['task_count']
    method = evaluation['methods'][diagnosis['arm']]
    direct = evaluation['methods']['direct']
    rows = []
    for index, (scored, control, label) in enumerate(zip(method['queries'], direct['queries'], labels)):
        path = Path(stage['run']) / diagnosis['arm'] / 'measured' / f'{index:03d}' / 'outcome.json'
        primary = read(path)['primary']
        assert primary['task_id'] == scored['task_id'] == control['task_id'] == label['task_id']
        assert primary['status'] == 'completed'
        units = {unit['id']: unit for unit in primary['sources']}
        support = set(label['evidence_document_ids'])
        frontiers = [set(units)]
        active = {reference['id']: set(units) for reference in primary['references']}
        for level in primary['levels']:
            outputs = [value for group in level['outputs'] for value in group]
            assert all(value['choice'] == NONE or value['choice'] in units for value in outputs)
            for field in {value['id'] for value in outputs}:
                active[field] = {value['choice'] for value in outputs
                                 if value['id'] == field and value['choice'] != NONE}
            frontiers.append(set().union(*active.values()))
        assert frontiers[-1] == {reference['source_id'] for reference in primary['references']
                                 if reference['source_id'] is not None}
        retention = []
        for frontier in frontiers:
            documents = {units[identity]['doc_id'] for identity in frontier}
            retention.append({'source_units': len(frontier), 'documents': len(documents),
                              'support_documents': len(support & documents),
                              'support_recall': len(support & documents) / len(support) if support else None})
        components = scored['component_seconds']
        rows.append({'task_id': primary['task_id'], 'question_type': label['question_type'],
                     'gold_document_count': len(support), 'retention': retention,
                     'workers': primary['worker_invocations'],
                     'exact_match': scored['scores']['primary']['exact_match'],
                     'direct_exact_match': control['scores']['primary']['exact_match'],
                     'elapsed_seconds': scored['elapsed_seconds'],
                     'direct_elapsed_seconds': control['elapsed_seconds'],
                     'component_seconds': components,
                     'residual_seconds': scored['elapsed_seconds'] - sum(components.values())})
    supported = [row for row in rows if row['gold_document_count']]
    summary = {'questions': len(rows), 'questions_with_gold_documents': len(supported),
               'mean_retrieved_document_recall': statistics.mean(row['retention'][0]['support_recall'] for row in supported),
               'mean_final_document_recall': statistics.mean(row['retention'][-1]['support_recall'] for row in supported),
               'retrieved_support_lost_in_selection': sum(row['retention'][0]['support_documents'] >
                                                         row['retention'][-1]['support_documents'] for row in supported),
               'component_seconds': method['component_seconds'],
               'total_seconds': method['elapsed_seconds'], 'direct_seconds': direct['elapsed_seconds']}
    report = {'stage': str(args.stage), 'scope': diagnosis['boundary'], 'summary': summary, 'queries': rows,
              'interpretation': 'Document-level retention does not establish passage-level sufficiency. Timings describe the executed method only.'}
    Path(diagnosis['output']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
