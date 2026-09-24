import argparse
from copy import deepcopy
import json
from pathlib import Path

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.reference_transport import pack_references, unpack_references
from jev_spawn.schema import CONTROLLER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    settings = json.loads(Path(config['transport']).read_text())
    shared = SharedConfig.load(config['shared_config'])
    method = json.loads(Path(config['method']).read_text())
    prompts = load_prompt(method['prompts'])
    transport_prompts = load_prompt(config['prompts'])
    typed_settings = json.loads(Path(config['typed_settings']).read_text())
    tokenizer = AutoTokenizer.from_pretrained(shared.model.path, local_files_only=True)
    marker = prompts['encoded_state'].format(value='')
    _, after_state = CONTROLLER['user_template'].split('{state}')
    question_prefix, after_question = after_state.split('{question}')
    menu_prefix, after_menu = after_question.split('{menu}')
    output_prefix, _ = after_menu.split('{output_instruction}')

    def serialize(value):
        return json.dumps(value, **settings['serialization'])

    def tokens(messages):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=shared.generation.enable_thinking)
        return tokenizer(rendered, add_special_tokens=False)['input_ids']

    records = []
    for source in config['sources']:
        for index, failure in enumerate(json.loads(Path(source['path']).read_text())):
            original_messages = failure['messages']
            assert len(tokens(original_messages)) == failure['input_tokens']
            messages = deepcopy(original_messages)
            user, = [message for message in messages if message['role'] == 'user']
            begin = user['content'].index(marker)
            start = begin + len(marker)
            payload, stop = json.JSONDecoder().raw_decode(user['content'][start:])
            tail = user['content'][start + stop:]
            assert tail.startswith(question_prefix)
            question, menu_and_end = tail[len(question_prefix):].split(menu_prefix)
            menu, end = menu_and_end.rsplit(output_prefix, config['split_count'])
            candidates, labels = [], []
            for candidate_index, line in enumerate(menu.splitlines()):
                label, description = line.split(config['label_separator'], config['split_count'])
                labels.append(label)
                if source['candidate_encoding'] == 'typed_json':
                    value = json.loads(description)
                    assert json.dumps(value, ensure_ascii=False) == description
                    candidate = {'id': typed_settings['option_id'].format(index=candidate_index), 'value': value}
                else:
                    option = prompts[source['options_key']][candidate_index]
                    assert option['description'] == description
                    candidate = dict(option)
                candidates.append(candidate)
            original = {'input': payload, 'candidates': candidates}
            packed = pack_references(original, settings)
            restored = unpack_references(packed, settings)
            assert serialize(restored) == serialize(original)
            packed_raw = pack_references(payload, settings)
            assert serialize(unpack_references(packed_raw, settings)) == serialize(payload)
            reference_menu = '\n'.join(
                label + config['label_separator'] + transport_prompts['reference_option'].format(index=offset)
                for offset, label in enumerate(labels))
            user['content'] = (user['content'][:begin]
                               + transport_prompts['reference_state'].format(value=serialize(packed))
                               + question_prefix + question + menu_prefix + reference_menu + output_prefix + end)
            records.append({'source': source['path'], 'index': index, 'task_id': failure['task_id'],
                            'original_tokens': failure['input_tokens'], 'packed_tokens': len(tokens(messages)),
                            'candidate_count': len(candidates), 'dictionary_entries': len(packed[settings['dictionary_key']]),
                            'exact_typed_reconstruction': True, 'raw_payload_roundtrip': True})
    result = {'scope': 'CPU-only reversible transport and actual tokenizer admission. No model quality or speed claim.',
              'candidate_recovery': 'Saved typed menus invert the known TypedValues JSON serialization and indexed IDs; ordinary mask options are verified against configured literal descriptions. Production must supply actual candidate values directly.',
              'original_background_and_question_unchanged': True,
              'records': records, 'requests': len(records), 'max_input_tokens': shared.model.max_input_tokens,
              'all_fit': all(row['packed_tokens'] <= shared.model.max_input_tokens for row in records)}
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'requests': len(records), 'all_fit': result['all_fit'],
                      'maximum_packed_tokens': max(row['packed_tokens'] for row in records)}))


if __name__ == '__main__':
    main()
