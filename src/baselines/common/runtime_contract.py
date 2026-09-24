from copy import deepcopy
import json

from jev_spawn.infra.prompts import load_prompt


class RuntimeContract:
    def configure_runtime_contract(self, method_settings):
        contract = {
            'max_turns': self.shared.runtime.max_turns,
            'max_new_tokens': self.shared.generation.max_new_tokens,
            'max_input_tokens': self.shared.model.max_input_tokens,
            'sample_timeout_seconds': self.shared.runtime.sample_timeout_seconds,
            'candidate_capacity': len(self.backend.answer_labels),
        }
        if 'rollout' in method_settings:
            contract.update({name: method_settings['rollout'][name]
                             for name in ('branch_width', 'parent_width')})
        self.runtime_contract = load_prompt('shared.runtime_contract').format(contract=json.dumps(contract))
        self.execution_metadata['runtime_contract'] = contract

    def contract_messages(self, messages):
        messages = deepcopy(messages)
        systems = [message for message in messages if message['role'] == 'system']
        if systems:
            systems[0]['content'] += '\n\n' + self.runtime_contract
        else:
            messages.insert(0, {'role': 'system', 'content': self.runtime_contract})
        return messages

