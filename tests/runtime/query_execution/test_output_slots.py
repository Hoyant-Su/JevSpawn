import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from jev_spawn.runtime.output_slots import OutputSlots, fill_slots, assemble_slots
from jev_spawn.infra.prompts import load_prompt


ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((ROOT / 'configs/jevspawn/output_slots.json').read_text())
FIXTURE = json.loads(Path(__file__).with_name('transport.json').read_text())


class OutputSlotTests(unittest.TestCase):
    def test_runtime_allocates_and_binds_nested_positions(self):
        output = OutputSlots(SETTINGS)
        expected = FIXTURE['assembled_answer']
        slots = output.object(output.root, list(expected))
        by_key = dict(zip(expected, slots, strict=True))
        array = output.array(by_key.pop('evidence'), len(expected['evidence']))
        for identity, value in zip(array, expected['evidence'], strict=True):
            output.assign(identity, value)
        for key, identity in by_key.items():
            output.assign(identity, expected[key])
        self.assertEqual(output.result(), expected)
        self.assertEqual(output.frontier(), [])

    def test_fill_preserves_native_values_and_batches_distinct_open_slots(self):
        case = FIXTURE['slot_readout_case']
        service = Mock()
        service.decide.return_value = case['decisions']
        service.complete_batch.return_value = case['texts']
        result = fill_slots(case['query'], case['state'], case['slots'], service=service,
                            task_id=FIXTURE['task_id'], prompts=load_prompt('jevspawn.output_slots'),
                            budget=case['budget'])
        self.assertEqual(result, case['expected'])
        service.decide.assert_called_once()
        service.complete_batch.assert_called_once()
        messages, *_ = service.complete_batch.call_args.args
        open_slots = [slot for slot in case['slots'] if not slot['options']]
        self.assertEqual(len(messages), len(open_slots))
        self.assertNotEqual(*messages)
        output = OutputSlots(SETTINGS)
        for identity, key in zip(output.object(output.root, list(result)), result, strict=True):
            output.assign(identity, result[key])
        self.assertEqual(json.loads(json.dumps(output.result())), case['expected'])

    def test_assembly_owns_structure_and_reuses_native_reference(self):
        case = FIXTURE['assembly_slot_case']
        service = Mock()
        service.decide.side_effect = case['decisions']
        service.complete_batch.return_value = case['texts']
        trace = []
        value = assemble_slots(case['query'], case['state'], case['known_values'],
            service=service, task_id=FIXTURE['task_id'], settings=SETTINGS,
            prompts=load_prompt(SETTINGS['prompts']), budget=case['budget'], trace=trace, schema={})
        self.assertEqual(value, case['expected'])
        service.complete_batch.assert_called_once()
        self.assertEqual(len(trace), len(case['decisions']))
        self.assertEqual(json.loads(json.dumps(value)), case['expected'])

    def test_declared_keys_and_types_need_no_model_structure_tokens(self):
        case = FIXTURE['slot_readout_case']
        opened = [slot for slot in case['slots'] if not slot['options']]
        schema = {'type': SETTINGS['object_kind'], 'properties': {
            slot['id']: {'type': SETTINGS['string_kind']} for slot in opened}}
        service = Mock()
        service.complete_batch.return_value = case['texts']
        trace = []
        result = assemble_slots(case['query'], case['state'], {}, service=service,
            task_id=FIXTURE['task_id'], settings=SETTINGS, prompts=load_prompt(SETTINGS['prompts']),
            budget={**case['budget'], 'max_turns': FIXTURE['assembly_slot_case']['budget']['max_turns']},
            trace=trace, schema=schema)
        service.decide.assert_not_called()
        service.complete_batch.assert_called_once()
        self.assertEqual(result, {slot['id']: value for slot, value in zip(opened, case['texts'], strict=True)})

    def test_dependent_values_see_subjects_while_records_remain_batched(self):
        case = FIXTURE['slot_dependency_case']
        service = Mock()
        service.complete_batch.side_effect = case['outputs']
        trace = []
        result = assemble_slots(case['query'], case['state'], {}, service=service,
            task_id=FIXTURE['task_id'], settings=SETTINGS, prompts=load_prompt(SETTINGS['prompts']),
            budget=case['budget'], trace=trace, schema=case['schema'])
        self.assertEqual(result, case['expected'])
        service.decide.assert_not_called()
        batches = [call.args[0] for call in service.complete_batch.call_args_list]
        self.assertEqual([len(batch) for batch in batches], [len(batch) for batch in case['outputs']])
        for messages in batches[-1]:
            for record in case['expected']:
                self.assertIn(record['subject'], messages[-1]['content'])

    def test_unresolved_position_is_not_a_null_answer(self):
        output = OutputSlots(SETTINGS)
        with self.assertRaises(ValueError):
            output.result()

    def test_duplicate_keys_preserve_pending_slot(self):
        output = OutputSlots(SETTINGS)
        before = output.frontier()
        with self.assertRaises(ValueError):
            output.object(output.root, [SETTINGS['root_key'], SETTINGS['root_key']])
        self.assertEqual(output.frontier(), before)


if __name__ == '__main__':
    unittest.main()
