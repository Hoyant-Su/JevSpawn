import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

from baselines.hiagent.adapter import DocumentEnvironment, original_agent


class LengthModel:
    def __init__(self, tokenizer, settings):
        self.tokenizer = tokenizer
        self.context_length = settings['context_length']
        self.max_tokens = settings['max_new_tokens']
        self.engine = 'qwen3.5-4b'

    def num_tokens_from_messages(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    prompts = json.loads(Path(settings['prompts']).read_text())
    native = json.loads(Path(settings['native_config']).read_text())
    collections = [json.loads(line) for line in Path(settings['collections']).read_text().splitlines()]
    assert len(collections) == settings['task_count'] == 104
    tokenizer = AutoTokenizer.from_pretrained(native['model_path'], local_files_only=True)
    model = LengthModel(tokenizer, settings)
    hiagent = settings['adapter_module'] == 'baselines.hiagent.adapter'
    environment_prompts = prompts if hiagent else prompts['environment']
    records = []
    os.environ['EVALTASK'] = 'bright_pony'
    for collection in collections:
        assert len(collection['candidates']) == settings['candidate_count'] == 128
        environment = DocumentEnvironment(collection, settings, environment_prompts)
        catalog = environment.reset()
        if hiagent:
            agent = original_agent(settings['source_directory'])(model,
                memory_size=settings['memory_size'], instruction=prompts['instruction'],
                examples=[], system_message=prompts['system'], need_goal=True,
                check_actions=prompts['check_actions'], use_parser=True)
            agent.reset(collection['query'], catalog)
            user = agent.make_prompt(need_goal=agent.need_goal, check_actions=agent.check_actions,
                                    check_inventory=agent.check_inventory, system_message=prompts['system'])
        else:
            user = collection['query'] + '\n\n' + catalog
        assert catalog in user and collection['query'] in user
        tokens = model.num_tokens_from_messages([
            {'role': 'system', 'content': prompts['system']}, {'role': 'user', 'content': user}])
        assert tokens + settings['max_new_tokens'] <= settings['context_length']
        records.append({'task_id': collection['task_id'], 'initial_input_tokens': tokens,
                        'candidate_count': len(environment.documents)})
    result = {'scope': 'CPU tokenization of complete original initial prompts. No model inference or relevance-label access.',
              'tasks': len(records), 'max_initial_input_tokens': max(row['initial_input_tokens'] for row in records),
              'max_new_tokens': settings['max_new_tokens'], 'context_length': settings['context_length'],
              'full_catalog_and_query_preserved': True, 'records': records}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}))


if __name__ == '__main__':
    main()
