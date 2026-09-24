import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT.parents[1] / 'data/research_v1'
OUTPUT = ROOT / 'baselines/model_capacity'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    selection = {'aqua': 32, 'supergpqa': 16, 'medxpertqa_text': 5}
    tasks, labels = [], []
    for dataset, count in selection.items():
        source = DATA / dataset / 'feasibility'
        selected = rows(source / 'tasks.jsonl')[:count]
        assert len(selected) == count
        ids = {row['task_id'] for row in selected}
        assert not ids & {row['task_id'] for row in rows(DATA / dataset / 'evaluation/tasks.jsonl')}
        tasks.extend(selected)
        gold = {row['task_id']: row for row in rows(source / 'labels.jsonl')}
        labels.extend(gold[row['task_id']] for row in selected)
    warm = rows(DATA / 'aqua/feasibility/tasks.jsonl')[32:40]
    assert len(warm) == 8 and not {row['task_id'] for row in tasks} & {row['task_id'] for row in warm}
    for name, values in [('tasks', tasks), ('labels', labels), ('warmup', warm)]:
        (OUTPUT / (name + '.jsonl')).write_text(''.join(json.dumps(row) + '\n' for row in values))
    settings = []
    for size in ['4B', '27B']:
        native = read(ROOT / 'configs/native.json')
        model = {'4B': 'Qwen3.5-4B', '27B': 'Qwen3.8-27B'}[size]
        native['model_path'] = '/inspire/hdd/global_public/public_models/Qwen/' + model
        save(OUTPUT / ('native-' + size + '.json'), native)
        for method, template in [('single', 'configs/baselines/single_reasoning/configs/aqua.json'),
                                 ('react', 'configs/baselines/formal_choices/configs/aqua_react.json')]:
            config = read(ROOT / template)
            config.update(dataset='development_capacity', task_count=len(tasks),
                          tasks=str(OUTPUT / 'tasks.jsonl'), labels=str(OUTPUT / 'labels.jsonl'),
                          warmup_tasks=str(OUTPUT / 'warmup.jsonl'),
                          warmup_task_ids=[row['task_id'] for row in warm],
                          native_config=str(OUTPUT / ('native-' + size + '.json')),
                          task_scope='Fixed development-only capacity comparison, preserving full original options.',
                          task_adaptation='Unchanged direct reasoning and original ReAct algorithms. Only model size and development inputs differ from the existing configurations.')
            path = OUTPUT / (method + '-' + size + '.json')
            save(path, config)
            settings.append(str(path.relative_to(ROOT)))
    save(OUTPUT / 'stage.json', {
        'stage': 'Backbone capacity before complete baseline matrix', 'status': 'prepared',
        'selection': 'First32 AQuA development, all16 prepared SuperGPQA development and all5 prepared MedXpertQA development, independent of observed scores. AQuA development positions32to39 provide disjoint warmup.',
        'task_count': len(tasks), 'by_dataset': selection, 'settings': settings,
        'fixed': {'dtype': 'bfloat16', 'hardware': 'One H10080GB per arm', 'input_tokens': 8192,
                  'model_batch_capacity': 8, 'root_batch_capacity': 8, 'seed': 0},
        'methods': ['Original ReAct', 'Single reasoning control'],
        'comparison': 'Use the user-specified Qwen3.8-27B directory against Qwen3.5-4B. This changes checkpoint identity and size, not necessarily size alone. Retain identical method settings and report exact model paths. No small-development result establishes full benchmark performance.',
        'decision': 'Prefer27B for the common primary matrix if it matches or improves pooled accuracy for both policies and improves at least one. Report paired uncertainty, per-domain scores, completion, memory and decoding tails. If inconclusive, retain the uncertainty rather than claiming4B sufficient.',
        'pre_run_review': 'All original inputs and options retained. No evaluation identities overlap. Labels are read only by preparation and offline scoring, and are absent from model task records. Native modelB8 and rootB8 match both sizes; budgets and original algorithms are identical within each pair. Interrupted blocks resume only after original writer termination. All failures remain in53assigned tasks.'})
    print(json.dumps({'tasks': len(tasks), 'warmup': len(warm), 'settings': settings}))


if __name__ == '__main__':
    main()
