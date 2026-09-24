from copy import deepcopy
import json
from pathlib import Path
import statistics

from transformers import AutoTokenizer
import yaml

from baselines.common.environment import TaskEnvironment
from baselines.common.schema_environment import environment_definition
from baselines.common.tasks import rows
from baselines.latentmas.common_service import role_messages
from methods.evidence_flow.environment import EvidenceEnvironment


FIXTURE = json.loads(Path('tests/baselines/common/fixtures/display_contract_measurement.json').read_text())


def summary(rows, key):
    values = [row[key] for row in rows]
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def main():
    specification = json.loads(Path(FIXTURE['specification']).read_text())
    factory, definition = environment_definition(specification)
    latent_run, compiler_run = Path(FIXTURE['latentmas_run']), Path(FIXTURE['llmcompiler_run'])
    latent_protocol = json.loads((latent_run / 'protocol.json').read_text())
    compiler_protocol = json.loads((compiler_run / 'protocol.json').read_text())
    shared = yaml.safe_load(latent_protocol['shared_config_text'])
    tokenizer = AutoTokenizer.from_pretrained(shared['model']['path'], local_files_only=True)
    settings, prompts = latent_protocol['tools'], latent_protocol['prompts']
    corpus = EvidenceEnvironment(**settings['evidence'])

    def size(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=shared['generation']['enable_thinking'])
        return len(tokenizer(text, add_special_tokens=False)['input_ids'])

    def environments(task):
        options = dict(deadline=lambda: None, evidence=corpus)
        return (TaskEnvironment(task, settings, FIXTURE['tool_directory'], **options),
                factory(task, settings, FIXTURE['tool_directory'], **options))

    def initial(environment):
        return [{'role': 'system', 'content': prompts['system']},
                {'role': 'user', 'content': environment.reset()}]

    def latent_lengths(messages):
        return [size(role) for role in role_messages(messages, prompts, shared['model']['path'])]

    task_rows = []
    for task in rows(specification['tasks']):
        original, candidate = environments(task)
        old, new = initial(original), initial(candidate)
        old_roles, new_roles = latent_lengths(old), latent_lengths(new)
        task_rows.append({'task_id': task['task_id'], 'kind': task['kind'],
            'canonical_original': size(old), 'canonical_candidate': size(new),
            'original_roles': old_roles, 'candidate_roles': new_roles,
            'role_input_total_original': sum(old_roles), 'role_input_total_candidate': sum(new_roles)})

    failures = {row['task_id']: row for row in json.loads((latent_run / FIXTURE['failures']).read_text())}
    compiler_messages = {}
    for batch in json.loads((compiler_run / FIXTURE['batches']).read_text()):
        for identity, messages in zip(batch['task_ids'], batch['messages'], strict=True):
            compiler_messages.setdefault(identity, messages)
    bright_rows = []
    assert latent_protocol['tasks'] == compiler_protocol['tasks']
    for task in latent_protocol['tasks']:
        original, candidate = environments(task)
        saved = failures[task['task_id']]
        assert saved['messages'] == initial(original)
        old_roles, new_roles = latent_lengths(saved['messages']), latent_lengths(initial(candidate))
        assert old_roles == saved['role_input_tokens']
        unchanged_latent_tokens = saved['required_context_tokens'] - sum(old_roles)
        old_compiler = compiler_messages[task['task_id']]
        question = original.context(False, {})
        assert sum(message['content'].count(question) for message in old_compiler) == 1
        new_compiler = [{**message, 'content': message['content'].replace(question, candidate.context(False, {}))}
                        for message in old_compiler]
        bright_rows.append({'task_id': task['task_id'], 'catalog_entries': len(task['input']['catalog']),
            'latentmas_canonical_original': size(saved['messages']), 'latentmas_canonical_candidate': size(initial(candidate)),
            'latentmas_original_roles': old_roles, 'latentmas_candidate_roles': new_roles,
            'latentmas_required_original': saved['required_context_tokens'],
            'latentmas_required_candidate': sum(new_roles) + unchanged_latent_tokens,
            'llmcompiler_planner_original': size(old_compiler), 'llmcompiler_planner_candidate': size(new_compiler)})
    report = {'stage': 'Isolated display contract CPU qualification; no GPU inference or active run activation',
        'specification': FIXTURE['specification'], 'environment_execution': definition,
        'model_tokenizer': shared['model']['path'], 'task_rows': task_rows, 'bright_rows': bright_rows,
        'legacy_rendering': 'Pre-edit captured full reset, exclude-finish, and Unicode exclude-finish prompts are byte-identical for all 14 actual tasks. Eight configured factories use the same exact interface projection.',
        'preservation': 'All input values, answer constraints, tool signatures and input schemas reconstruct exactly; actual search, read and all BRIGHT document observations are unchanged. Runtime validation uses original schemas.',
        'scope': 'CPU rendering and information-preservation checks only. This is not a full execution of eight method loops or a performance/accuracy result. LatentMAS uses genuine role_messages construction and checks every original role length against its saved rejection. LLMCompiler replaces the unique actual initial question inside its saved full planner request. Later tool histories and method context policies are unchanged.',
        'source_runs': [str(latent_run), str(compiler_run)], 'max_input_tokens': shared['model']['max_input_tokens'],
        'consumer_boundary': 'DyFlow standalone TOOL_CALL and ORGANIZE_SOLUTION schemas remain complete because upstream selects operator input_keys and may omit original_problem. Other full-context paths use the candidate contract; HiAgent check-actions references are scoped to its persistent goal (need_goal=True).',
        'tests': ['legacy_rendering_is_byte_identical_for_every_real_task',
                  'original_eight_formal_protocols_serialize_identically',
                  'original_schema_refs_and_literal_reference_keys_are_not_reinterpreted',
                  'all_method_factories_preserve_complete_contracts_for_every_task',
                  'real_search_read_observations_and_private_validation_are_unchanged',
                  'real_catalog_ids_and_document_observations_are_unchanged']}
    keys = ('latentmas_canonical_original', 'latentmas_canonical_candidate', 'latentmas_required_original',
            'latentmas_required_candidate', 'llmcompiler_planner_original', 'llmcompiler_planner_candidate')
    report['bright_summary'] = {key: summary(bright_rows, key) for key in keys}
    for key in ('latentmas_required_original', 'latentmas_required_candidate',
                'llmcompiler_planner_original', 'llmcompiler_planner_candidate'):
        report['bright_summary'][key]['over_input_limit'] = sum(row[key] > report['max_input_tokens'] for row in bright_rows)
    report['promotion'] = {'decision': 'Not promoted as a universal display encoding',
        'reason': 'Reference overhead increases most small-task prompts and introduces an additional LatentMAS input-limit failure on the SQuAD2 qualification example. All full BRIGHT LatentMAS tasks still exceed the unchanged context limit.',
        'qualification_canonical_increases': sum(row['canonical_candidate'] > row['canonical_original'] for row in task_rows),
        'qualification_canonical_decreases': sum(row['canonical_candidate'] < row['canonical_original'] for row in task_rows),
        'activation': 'Explicit candidate configs only; original formal configurations and protocols unchanged.'}
    Path(FIXTURE['destination']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['bright_summary'], indent=2))


if __name__ == '__main__':
    main()
