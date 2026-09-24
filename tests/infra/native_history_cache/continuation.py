from collections import OrderedDict, defaultdict

from baselines.common.jevspawn_service import StructuredService
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.schema import CONTROLLER


class SelectedPrefixes:
    def __init__(self, base_cache, selected):
        self.base_cache, self.selected = base_cache, selected

    def get_many(self, prefixes, prefill):
        missing = [prefix for prefix in prefixes if tuple(prefix) not in self.selected]
        states, hits = self.base_cache.get_many(missing, prefill)
        values = {tuple(prefix): (state, hit) for prefix, state, hit in zip(missing, states, hits, strict=True)}
        values.update({key: (state, True) for key, state in self.selected.items()})
        return ([values[tuple(prefix)][0] for prefix in prefixes],
                [values[tuple(prefix)][1] for prefix in prefixes])


def continue_messages(anchor, field, labels, prompts):
    menu = '\n'.join(CONTROLLER['option_template'].format(label=label, **option)
                     for label, option in zip(labels, field['options']))
    content = prompts['request'].format(state=field['state'], question=field['question'],
        menu=menu, output_instruction=CONTROLLER['output_instruction'])
    return [*anchor['messages'], {'role': 'assistant', 'content': anchor['label']},
            {'role': 'user', 'content': content}]


class ContinuationTail(StableFiniteGraphTail):
    def __init__(self, *args, continuation):
        super().__init__(*args)
        self.anchor_states = OrderedDict()
        self.continuation = continuation

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        groups = defaultdict(list)
        for row, request in enumerate(requests):
            groups[(request.task_id, request.field['context'])].append(row)
        lengths, retained, reused = list(base_lengths), {}, []
        for indices in groups.values():
            sequences = [requests[row].admitted.tokens for row in indices]
            boundary = min(common_prefix(sequences), min(map(len, sequences)) - 1)
            available = [value for key, value in self.anchor_states.items()
                if key[0] == requests[indices[0]].task_id and len(value[0]) <= boundary
                and sequences[0][:len(value[0])] == value[0]]
            if available:
                prefix, state = max(available, key=lambda item: len(item[0]))
                retained[prefix] = state
                reused.append(len(prefix) - base_lengths[indices[0]])
                for row in indices:
                    lengths[row] = len(prefix)
        result = super().score(requests, lengths, SelectedPrefixes(base_cache, retained), physical_batch_size)
        result['reused_state_tokens'] = sum(reused)
        result['reused_root_tokens'] -= sum(reused)
        result['persistent_prefix_scope'] = 'task_root_and_completed_decision'
        for request in requests:
            if request.field['id'] in self.continuation['anchors']:
                prefix = tuple(request.admitted.tokens[:-1])
                identity = (request.task_id, request.field['id'])
                self.anchor_states[identity] = (prefix, self.prefix_cache.entries[(prefix,)])
                self.anchor_states.move_to_end(identity)
        while len(self.anchor_states) > self.runtime.root_batch_size * len(self.continuation['anchors']):
            self.anchor_states.popitem(last=False)
        return result


def install_messages(settings):
    original_init = StructuredService.__init__
    original_enqueue = StructuredService.enqueue_decisions
    prompts = load_prompt(settings['prompts'])

    def initialize(service, *args, **kwargs):
        original_init(service, *args, **kwargs)
        service.decision_anchors = {}

    def enqueue(service, fields, *, task_id, request_type):
        if any(field['id'] == settings['reset_field'] for field in fields):
            service.decision_anchors[task_id] = {}
        anchors = service.decision_anchors.setdefault(task_id, {})
        created = {}

        def construct(*args, **kwargs):
            request = request_type(*args, **kwargs)
            field = request.field
            parent = settings['parents'].get(field['id'], settings['field_parent'])
            anchor = anchors.get(parent)
            compatible = (anchor is not None and anchor['context'] == field['context']
                          and anchor['history'] == field['history'])
            if field['id'] != settings['reset_field'] and compatible:
                request.messages = continue_messages(anchor, field, service.backend.answer_labels, prompts)
                request.field['continuation_parent'] = parent
            created[field['id']] = request
            return request

        values = original_enqueue(service, fields, task_id=task_id, request_type=construct)
        for value in values:
            identity = value['id']
            if identity in settings['anchors'] and identity in created:
                request = created[identity]
                index = value['option_ids'].index(value['choice'])
                anchors[identity] = {'messages': request.messages, 'label': service.backend.answer_labels[index],
                    'context': request.field['context'], 'history': request.field['history']}
        return values

    StructuredService.__init__ = initialize
    StructuredService.enqueue_decisions = enqueue
