from pathlib import Path

from baselines.common.lats import EvaluatorPrompt
from baselines.common.tasks import read


def source_request(settings, prompts):
    source = read(Path(settings['source_run']) / 'session-0000/batches.json')[settings['source_batch_index']]
    index = source['task_ids'].index(settings['source_task_id'])
    message, = source['messages'][index]
    instruction = prompts['value_instruction']
    assert message['role'] == 'user' and message['content'].startswith(instruction)
    prompt = EvaluatorPrompt(instruction, message['content'][len(instruction):], prompts)
    assert str(prompt) == message['content']
    return prompt, {key: source[key][index] for key in
                    ['messages', 'texts', 'output_tokens', 'finish_reasons',
                     'requested_max_new_tokens', 'row_stops']}


def assessment(text, original_task):
    return {'raw_output': text, 'upstream_value': original_task.value_outputs_unwrap([text])}

