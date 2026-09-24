import json
import math
import os
from pathlib import Path

import jsonschema


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_result(task, result):
    assert result['task_id'] == task['task_id'], 'Committed task identity differs from the declared task.'
    assert result['status'] in {'completed', 'timeout', 'invalid_output', 'limit_exceeded'}, (
        'Committed task does not have a terminal lifecycle status.')
    assert math.isfinite(result['elapsed_seconds']) and result['elapsed_seconds'] >= 0
    if result['status'] == 'completed':
        if result['answer'] is not None:
            jsonschema.validate(result['answer'], task['answer_schema'])
    else:
        assert result['answer'] is None and isinstance(result['error'], str)


def pending_tasks(tasks, output):
    pending = []
    for index, task in enumerate(tasks):
        path = output / f'task-{index:05d}.json'
        if path.exists():
            validate_result(task, json.loads(path.read_text()))
        else:
            pending.append(task)
    return pending
