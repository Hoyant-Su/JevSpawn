from copy import deepcopy
import json


class ActionMessages:
    def __init__(self, messages, field, bindings, labels, prompts, controller, serialization):
        self.messages = messages
        self.pending = field
        self.bindings = dict(bindings)
        self.labels = labels
        self.prompts, self.controller, self.serialization = prompts, controller, serialization

    def fork(self):
        return ActionMessages(list(self.messages), self.pending, self.bindings, self.labels,
                              self.prompts, self.controller, self.serialization)

    def append(self, field, bindings, definitions, options):
        value = bindings[self.pending['id']]
        choice = self.pending['values'].index(value)
        selected = [{'field': definitions[identity], 'value': deepcopy(value)}
                    for identity, value in bindings.items() if identity not in self.bindings]
        menu = '\n'.join(self.controller['option_template'].format(label=label, **option)
                         for label, option in zip(self.labels, options))
        question = self.prompts['next_field'].format(
            bindings=json.dumps(selected, **self.serialization),
            field=json.dumps(field, **self.serialization), menu=menu,
            output_instruction=self.controller['output_instruction'])
        self.messages = [*self.messages, {'role': 'assistant', 'content': self.labels[choice]},
                         {'role': 'user', 'content': question}]
        self.pending, self.bindings = field, dict(bindings)
        return self.messages
