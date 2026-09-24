import argparse
from pathlib import Path

from transformers import AutoTokenizer

from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.evidence_flow.environment import read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.evidence_interfaces.interfaces import compact
from methods.hallucination_localization import schema
from methods.hallucination_localization.inputs import context, field, locate, partition, refine, words
from methods.structured_flow.grammar import SchemaDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = read(args.stage)
    fixed, prompts, operators = stage['fixed'], read(stage['prompts']), read(stage['operators'])
    tokenizer = AutoTokenizer.from_pretrained(fixed['model_path'])
    config = read(Path(fixed['model_path']) / 'config.json')['text_config']
    decoder = SchemaDecoder(tokenizer, [config['eos_token_id']], config['vocab_size'])
    CONTROLLER['option_template'] = prompts['option_template']
    rows = []
    for task in read_jsonl(stage['data']['tasks']):
        units = words(task['response'])
        contract = schema.partition(len(units))
        decoder.factory(contract, kind='answer')
        decoder.factory(schema.spans(), kind='answer')
        root, = partition([len(units)], len(units))
        pending, leaves = [root], []
        while pending:
            node = pending.pop()
            if node['end'] - node['start'] == 1:
                leaves.append(node)
            else:
                pending.extend(refine(node))
        assert sorted((node['start'], node['end']) for node in leaves) == [(i, i + 1) for i in range(len(units))]
        assert locate(task['response'], [{'text': task['response'], 'occurrence': 1}]) == [[0, len(task['response'])]]
        indexed = compact([[i, task['response'][start:end]] for i, (start, end) in enumerate(units)])
        planner = prompts['planner_user'].format(state=context(task), words=indexed, count=len(units), schema=compact(contract))
        direct = prompts['direct_user'].format(state=context(task), schema=compact(schema.spans()))
        item = field(root, task, units, prompts, operators)
        letters = [chr(65 + i) for i in range(len(item['options']))]
        worker = controller_prompts([item['state']], item['question'], item['options'], letters,
                                    prompts['json_instruction'])[0]
        lengths = []
        for system, prompt in [(prompts['planner_system'], planner), (prompts['direct_system'], direct),
                               (CONTROLLER['system'], worker)]:
            text = tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}],
                                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
            lengths.append(len(tokenizer(text, add_special_tokens=False)['input_ids']))
        assert max(lengths) <= fixed['context_tokens'], (task['task_id'], lengths)
        rows.append({'task_id': task['task_id'], 'words': len(units), 'input_tokens': lengths})
    assert len(rows) == stage['data']['task_count']
    write(Path(stage['cpu_preflight']), {'tasks': rows,
          'scope': 'Complete real development text, exact recursive interval coverage and original-context token lengths. Subdivision check is not an inferred model trajectory.'})
    print(rows)


if __name__ == '__main__':
    main()
