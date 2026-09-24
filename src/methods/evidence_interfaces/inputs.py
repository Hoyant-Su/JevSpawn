import json
from pathlib import Path
from jev_spawn.infra.prompts import resolve_prompts


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def inputs(settings):
    stage = resolve_prompts(read(settings['stage']))
    limits = stage['fixed']
    rows = [json.loads(line) for line in Path(stage['data']['collections']).read_text().splitlines()]
    assert len(rows) == stage['data']['queries']
    assert len({row['task_id'] for row in rows}) == len(rows)
    for row in rows:
        assert len(row['candidates']) == stage['data']['documents_per_query']
        assert len({doc['document_id'] for doc in row['candidates']}) == len(row['candidates'])
        assert all(doc['text'] and isinstance(doc['text'], str) for doc in row['candidates'])
    assert settings['ranking_length'] == limits['ranking_cutoff'] == limits['ranking_length']
    assert settings['worker_tokens'] == limits['worker_tokens']
    assert settings['require_eos'] and limits['measured_repetitions'] == limits['root_tasks_in_flight'] == limits['gpu_count'] == 1
    assert limits['max_rounds'] == 2 and limits['max_options_per_field'] == len(settings['option_letters'])
    return stage, rows


