import argparse
from copy import deepcopy
import json

from transformers import AutoTokenizer

from jev_spawn.runtime.structured_messages import judgment_table
from project_paths import ROOT


def run(settings):
    failures = json.loads((ROOT / settings['source']).read_text())
    failure = failures[settings['failure_index']]
    prompts = json.loads((ROOT / settings['prompts']).read_text())['templates'][settings['prompt_group']]
    policy = json.loads((ROOT / settings['policy']).read_text())
    text = failure['messages'][settings['user_message_index']]['content']
    prefix, body = text.split(settings['state_start'])
    original_state, suffix = body.split(settings['state_end'])
    state = json.loads(original_state)
    references = [*state['observations'], *(item['result'] for item in state['observations']),
                  state['focus'], *state['focus']['children']]
    descriptions = []
    for reference in references:
        description = prompts['evidence_reference'].format(
            path=json.dumps(reference['reference']), kind=reference['kind'], count=reference['count'])
        assert reference['description'] == description
        descriptions.append(reference.pop('description'))
    original_judgments = state['judgments']
    state['judgments'] = judgment_table(original_judgments, policy['communication'])
    table = state['judgments']
    decoded = [dict(zip(table['columns'], row, strict=True)) for row in table['rows']]
    for judgment in decoded:
        judgment['bindings'] = [dict(zip(table['binding_columns'], row, strict=True))
                               for row in judgment['bindings']]
    assert decoded == original_judgments
    compact = prompts['encoded_state'].format(value=json.dumps(state, **policy['serialization']))
    messages = deepcopy(failure['messages'])
    messages[settings['user_message_index']]['content'] = (
        prefix + settings['state_start'] + compact + settings['state_end'] + suffix)
    tokenizer = AutoTokenizer.from_pretrained(settings['model'], local_files_only=True)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=settings['enable_thinking'])
    tokens = tokenizer(rendered, add_special_tokens=False)['input_ids']
    state['judgments'] = decoded
    for reference, description in zip(references, descriptions, strict=True):
        reference['description'] = description
    assert state == json.loads(original_state)
    result = {'source': settings['source'], 'original_tokens': failure['input_tokens'],
              'compact_tokens_without_option_shortening': len(tokens),
              'max_input_tokens': failure['max_input_tokens'],
              'reference_descriptions_exactly_reconstructible': len(references),
              'original_state_reconstructed_exactly': True, 'scope': settings['scope']}
    (ROOT / settings['output']).write_text(json.dumps(result, indent=settings['json_indent']) + '\n')
    print(json.dumps(result, indent=settings['json_indent']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('settings')
    arguments = parser.parse_args()
    run(json.loads((ROOT / arguments.settings).read_text()))
