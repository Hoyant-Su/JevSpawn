import argparse
import json
from pathlib import Path
import time

import jsonschema
import numpy as np
from transformers import AutoTokenizer

from jev_spawn.schema import CONTROLLER
from methods.evidence_interfaces import schema
from methods.evidence_interfaces.interfaces import InvalidResponse, compact, evidence_view, fields, parsed, receiver_input, unique
from methods.evidence_interfaces.inputs import inputs, read, write
from methods.structured_flow.grammar import SchemaDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = read(args.settings)
    stage, rows = inputs(settings)
    limits, field = stage['fixed'], read(settings['field_schema'])
    prompts = read(settings['prompts'])
    model = Path(limits['model_path'])
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    text_config = read(model / 'config.json')['text_config']
    eos = text_config['eos_token_id']
    eos_ids = eos if isinstance(eos, list) else [eos]
    decoder = SchemaDecoder(tokenizer, eos_ids, text_config['vocab_size'])
    count, cutoff = stage['data']['documents_per_query'], settings['ranking_length']
    question = {'question': 'CPU grammar fixture question', 'options': ['Supported', 'Not supported']}
    fixtures = [
        (schema.planner(field, limits), {'questions': [question]}),
        (schema.coordinator(field, limits, count, cutoff), {'action': 'finish', 'ranking': list(range(cutoff))}),
        (schema.coordinator(field, limits, count, cutoff), {'action': 'refine', 'documents': list(range(count)), 'questions': [question]}),
        (schema.ranking(count, cutoff), {'ranking': list(range(cutoff))}),
    ]
    started, tokens = time.perf_counter(), 0
    for contract, value in fixtures:
        jsonschema.validate(value, contract)
        constraint = decoder.factory(contract, kind='answer')(tokenizer, eos_ids, 0, limits['coordinator_tokens'])
        encoded = tokenizer(compact(value), add_special_tokens=False)['input_ids']
        for index, token in enumerate(encoded):
            assert token in constraint(0, np.array(encoded[:index], dtype=np.int64))
            tokens += 1
        assert eos_ids[0] in constraint(0, np.array(encoded))
        assert not set(eos_ids) & set(constraint(1, np.array([], dtype=np.int64)))
    typed = fields([question], 0, [])
    cases = [lambda: fields([question, question], 0, []),
             lambda: fields([question], 1, typed), lambda: unique([1, 1], 'document IDs'),
             lambda: parsed({'result': {'truncated': [True], 'texts': ['{}']}}, fixtures[0][0])]
    for check in cases:
        try:
            check()
        except InvalidResponse:
            pass
        else:
            raise AssertionError('Invalid model response accepted')
    messages = [{'r0q0': 'A'} for _ in range(count)]
    received = receiver_input(rows[0], typed, messages, fixtures[-1][0], settings, prompts)
    assert all(doc['text'] not in received for doc in rows[0]['candidates'])
    assert len(evidence_view(typed, messages)['items']) == count
    second = fields([{'question': 'Different CPU fixture question', 'options': ['Present', 'Absent']}], 1, typed)
    assert evidence_view(typed + second, messages)['items'][0] == [0, ['A', None]]
    CONTROLLER['option_template'] = prompts['option_template']
    documents = [doc for row in rows for doc in row['candidates']]
    lengths = tokenizer([doc['text'] for doc in documents], add_special_tokens=False)['input_ids']
    assert max(map(len, lengths)) < limits['context_tokens']
    output = {'model_loaded': False, 'grammar_fixtures': len(fixtures), 'real_tokenizer_tokens_checked': tokens,
              'invalid_responses_rejected': len(cases), 'receiver_excludes_document_text': True,
              'queries': len(rows), 'documents': len(documents), 'max_raw_document_tokens': max(map(len, lengths)),
              'grammar_and_structure_seconds': time.perf_counter() - started,
              'scope': 'CPU schema, token-constraint and data-path validation. No model inference, predictions, or task metrics.'}
    write(args.output, output)
    print(json.dumps(output))


if __name__ == '__main__':
    main()
