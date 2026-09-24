from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from baselines.common.config import SharedConfig


SOURCE = Path('runs/common-jevspawn-candidate-feedback-tp4-v2-002/session-0000/input_failures.json')
SHARED = SharedConfig.load('configs/shared_config_tp4_v2.yaml')
TOKENIZER = AutoTokenizer.from_pretrained(SHARED.model.path)


def size(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return {'characters': len(text), 'tokens': len(TOKENIZER(text, add_special_tokens=False)['input_ids'])}


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


failure, = json.loads(SOURCE.read_text())
user = failure['messages'][1]['content']
start = user.index('{"root"')
root, width = json.JSONDecoder().raw_decode(user[start:])
semantic = root['root']
rendered = TOKENIZER.apply_chat_template(failure['messages'], tokenize=False,
    add_generation_prompt=True, enable_thinking=SHARED.generation.enable_thinking)
actual = size(rendered)['tokens']
assert actual == failure['input_tokens']
counts = Counter(strings(semantic))
documents = [document for invocation in semantic['input']['observations'][0]['result']['value'] for document in invocation]
report = {
    'source': str(SOURCE), 'task_id': failure['task_id'], 'actual_input_tokens': actual,
    'limit': failure['max_input_tokens'],
    'parts_independently_tokenized_not_additive': {
        'system': size(failure['messages'][0]['content']), 'background_before_runtime': size(user[:start]),
        'literal_runtime': size(user[start:start + width]), 'question_and_option_labels': size(user[start + width:]),
        'runtime_input': {key: size(value) for key, value in semantic['input'].items()},
        'candidate_descriptions': size(semantic['candidates']),
        'candidate_groups': {kind: size([item for item in semantic['candidates'] if item['id'].startswith(kind)])
                             for kind in ('spawn:', 'inspect:', 'tool:')}},
    'documents': [{'id': item['document_id'], 'text_size': size(item['text']),
                   'exact_text_occurrences_in_semantic_runtime': counts[item['text']]} for item in documents],
    'catalog': {'options': len(semantic['candidates']), 'spawn': 35, 'inspect': 35, 'tools': 2, 'proposal': 1},
    'finding': 'All32 complete returned document texts occur once in the semantic runtime. Overflow is not caused by repeated document bodies in this failed controller call. The original root context remains, then newly acquired complete evidence plus73 expanded controller descriptions and their label menu add input length. Focus lists32 child descriptors, and the controller separately expands spawn and inspect choices for these same addresses.',
    'minimal_lossless_direction': 'Retain all evidence and all73 candidate identities. Expose the already available native action descriptors (operator, exact path, kind, count, and member shapes) and render repeated action description templates once with a literal table of candidate-specific values. Current finite_control source_description/solve discard these native descriptors into opaque repeated description strings. Reconstruct each original description byte-for-byte from its original template and literal fields. Do not parse or summarize document bodies, prune actions, or change selection policy. Full-prompt root answer schema and finish input_schema are also identical native objects that can share one explicit definition, but this is a separate transport boundary.',
    'prefix_layout': 'The literal service hoists configured evidence only when input has that key. This controller input directly contains observations/judgments/focus, so that hook hoists nothing. Existing immutable background remains first; new observation content is correctly after it. Prefix reuse avoids repeated computation but cannot make an over-limit logical request admissible.',
    'limits': 'Read-only diagnosis only. A compact candidate layout still needs exact reconstruction, tokenizer admission, and model-quality evaluation; no speed or quality improvement is inferred.'}
Path('results/infra/bright_candidate_feedback_payload_002.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'actual_tokens': actual, 'parts': report['parts_independently_tokenized_not_additive'],
                  'document_count': len(documents), 'body_occurrences': Counter(item['exact_text_occurrences_in_semantic_runtime'] for item in report['documents'])}, indent=2))
