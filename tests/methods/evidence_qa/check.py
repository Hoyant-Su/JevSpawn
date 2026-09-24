import ast
import json
from pathlib import Path

from transformers import AutoTokenizer

from methods.evidence_flow.environment import EvidenceEnvironment, read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.evidence_interfaces.interfaces import compact
from methods.evidence_qa import schema
from methods.structured_flow.grammar import SchemaDecoder


def main():
    stage = read('configs/methods/evidence_qa/stage.json')
    fixed = stage['fixed']
    for path in Path('methods/evidence_qa').rglob('*.py'):
        ast.parse(path.read_text())
    rows = read_jsonl(stage['data']['tasks'])
    environment = EvidenceEnvironment(stage['data']['corpus'], stage['data']['units'], **stage['environment'])
    tokenizer = AutoTokenizer.from_pretrained(fixed['model_path'], local_files_only=True)
    config = read(Path(fixed['model_path']) / 'config.json')['text_config']
    eos = config['eos_token_id']
    decoder = SchemaDecoder(tokenizer, eos if isinstance(eos, list) else [eos], config['vocab_size'])
    prompts, field = read(stage['prompts']), read(stage['field_schema'])
    for contract in [schema.interface(field, fixed), schema.receiver(field, fixed, 144), schema.answer(256)]:
        decoder.factory(contract, kind='answer')
    lengths = []
    for row in rows:
        episode = environment.episode(row['task_id'])
        hits = episode.search(row['query'], fixed['initial_search_count'])
        units = episode.read([hit['id'] for hit in hits])
        assert all(unit['text'] == environment.documents[unit['doc_id']]['body'][unit['start']:unit['end']] for unit in units)
        contract = schema.interface(field, fixed)
        prompt = prompts['planner_user'].format(query=row['query'], evidence=compact([{'index':i, **unit} for i,unit in enumerate(units)]),
             limits=compact({'fields':fixed['max_fields_per_round'],'options':fixed['max_options_per_field']}), schema=compact(contract))
        rendered = tokenizer.apply_chat_template([{'role':'system','content':prompts['planner_system']},
             {'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        length = len(tokenizer(rendered, add_special_tokens=False)['input_ids'])
        assert length <= fixed['context_tokens']
        lengths.append({'task_id':row['task_id'],'initial_passages':len(units),'planner_input_tokens':length})
    report={'tasks':lengths,'grammar_schemas_compiled':3,'corpus_documents':len(environment.documents),
            'source_units':len(environment.units),'source_offsets_preserved':True,'labels_read':False,'model_loaded':False}
    write(Path('results/methods/evidence_qa/cpu_preflight.json'), report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
