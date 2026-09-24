import argparse
import json
from pathlib import Path
import string

from transformers import AutoTokenizer

from jev_spawn.schema import CONTROLLER, controller_prompts


def prompt_lengths(tokenizer, states, fields):
    prompts = [controller_prompts([state], field['question'], field['options'],
                                 list(string.ascii_uppercase[:len(field['options'])]), CONTROLLER['output_instruction'])[0]
               for state, schema in zip(states, fields) for field in schema.values()]
    rendered = tokenizer.apply_chat_template([[{'role': 'system', 'content': CONTROLLER['system']},
                                               {'role': 'user', 'content': prompt}] for prompt in prompts],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return sum(map(len, tokenizer(rendered, add_special_tokens=False)['input_ids']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.run / 'protocol.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(protocol['native']['model_path'], local_files_only=True)
    CONTROLLER['option_template'] = protocol['prompts']['option_template']
    report = {'scope': 'CPU token accounting only. Typed messages are serialized as JSON text into frozen Qwen. No worker generates a rationale or free text summary. Identity only messages below are a counterfactual encoding, not a tested semantic substitute. Their meanings would require receiver access to the defining schemas.', 'phases': {}}
    for phase in ['warmup', 'measured']:
        path = args.run / phase
        result = {'receiver_states': 0, 'logical_full_prompt_tokens': 0,
                  'logical_empty_message_prompt_tokens': 0, 'logical_identity_message_prompt_tokens': 0,
                  'message_deliveries': 0, 'repeated_question_strings': 0, 'repeated_selected_description_strings': 0}
        decisions = json.loads((path / 'decision-calls.json').read_text())
        relevance = json.loads((path / 'relevance-calls.json').read_text())
        for record in decisions + relevance:
            states = record['states']
            if 'schemas' in record:
                fields = record['schemas']
            else:
                fields = [{'relevance': {'question': protocol['prompts']['relevance_question'],
                                         'options': protocol['prompts']['relevance_options']}} for _ in states]
            empty, identity = [], []
            for state in states:
                value = json.loads(state)
                messages = value['worker_messages']
                result['message_deliveries'] += len(messages)
                result['repeated_question_strings'] += sum('question' in message for message in messages)
                result['repeated_selected_description_strings'] += sum('description' in message for message in messages)
                empty.append(json.dumps({**value, 'worker_messages': []}, ensure_ascii=False))
                compact = [{key: item for key, item in message.items() if key not in ['question', 'description']}
                           for message in messages]
                identity.append(json.dumps({**value, 'worker_messages': compact}, ensure_ascii=False))
            result['receiver_states'] += len(states)
            result['logical_full_prompt_tokens'] += prompt_lengths(tokenizer, states, fields)
            result['logical_empty_message_prompt_tokens'] += prompt_lengths(tokenizer, empty, fields)
            result['logical_identity_message_prompt_tokens'] += prompt_lengths(tokenizer, identity, fields)
        result['logical_message_token_increment'] = result['logical_full_prompt_tokens'] - result['logical_empty_message_prompt_tokens']
        result['logical_question_description_token_increment'] = result['logical_full_prompt_tokens'] - result['logical_identity_message_prompt_tokens']
        report['phases'][phase] = result
    (args.run / 'message-token-accounting.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
