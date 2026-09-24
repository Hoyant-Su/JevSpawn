from pathlib import Path

from methods.evidence_interfaces.inputs import read
from methods.request_routing.analysis import hierarchy_statistics, validate_retrieval
from methods.source_interfaces.evaluate import evaluate, write_evaluation
from methods.source_sets.analysis import source_ids


if __name__ == '__main__':
    args, stage, report = evaluate(hierarchy_statistics, source_ids)
    for arm in stage['arms']:
        if arm == 'direct':
            continue
        for path in sorted((args.output / arm / 'measured').glob('*/outcome.json')):
            outcome = read(path)
            validate_retrieval(outcome['primary'], read(Path(outcome['attempt']) / 'tools.json'))
    reference = read(stage['quality_reference'])['methods']['streamed']
    ours = report['methods']['streamed']
    report['promotion']['dense_quality_preserved'] = (
        ours['exact_match'] >= reference['exact_match'] and ours['token_f1'] >= reference['token_f1'])
    report['promoted'] = all(report['promotion'].values())
    report['quality_reference'] = stage['quality_reference']
    write_evaluation(args, report)
