import json
import re

from jev_spawn.schema import DECISIONS, SYSTEM


def code_from_response(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.DOTALL)
    return '\n\n'.join(blocks).strip() if blocks else text.strip()


def observation(state, stage, review_focus=False):
    result = {'specification': state['prompt'], 'implementation': state['initial_solution']}
    if review_focus:
        result['review_focus'] = next(option for option in DECISIONS['review_focus']['options']
                                      if option['id'] == state['review_focus'])
    if stage in {'repair', 'select'}:
        result['review'] = state['review']
    if stage == 'select':
        result['revised_implementation'] = state['revised_solution']
    return json.dumps(result, ensure_ascii=False)


class Workflow:
    def __init__(self, backend, config, journal, rank):
        self.backend, self.config, self.journal, self.rank = backend, config, journal, rank
        assert config['field_mode'] in {'shared', 'independent'}
        assert config['controller_mode'] in {'direct', 'json'}

    def review_fields(self, states, events):
        adaptive = self.config['spawn_policy'] == 'adaptive'
        names = ['review', 'review_focus'] if adaptive else ['review_focus']
        mode = self.config['field_mode'] if adaptive else 'independent'
        observations = [observation(state, 'review') for state in states]
        fields = {name: DECISIONS[name] for name in names}
        if self.config['controller_mode'] == 'json':
            result = self.backend.generate_fields(observations, fields, self.config['controller_max_new_tokens'])
        else:
            result = self.backend.score_fields(observations, fields, mode=mode)
        events.append(self.journal.event(
            'batch_completed', role='controller', stage='review_fields', gpu_id=self.rank,
            batch_size=len(states), field_count=result['field_count'], field_mode=result['mode'],
            readout_mode=result['mode'],
            duration_seconds=result['elapsed_seconds'], input_tokens=result['computed_input_tokens'],
            logical_input_tokens=result['logical_input_tokens'],
            computed_input_tokens=result['computed_input_tokens'], padded_input_tokens=result['padded_input_tokens'],
            output_tokens=sum(result['output_tokens']), timings=result['timings'],
        ))
        for index, state in enumerate(states):
            state['review_focus'] = result['fields']['review_focus']['choices'][index]
            for name in names:
                answer = result['fields'][name]
                decision = {
                    'stage': name, 'choice': answer['choices'][index], 'options': DECISIONS[name]['options'],
                    'probabilities': answer['probabilities'][index], 'option_logits': answer['option_logits'][index],
                    'input_tokens': answer['input_tokens'][index],
                    'input_token_semantics': ('Joint prompt length; count once across fields.' if result['mode'] == 'joint_json'
                                             else 'Logical field prompt length before prefix sharing.'),
                    'probability_status': ('Not available from generated JSON.' if result['mode'] == 'joint_json'
                                           else 'Uncalibrated conditional option scores.'),
                }
                state['decisions'].append(decision)
                events.append(self.journal.event('decision', task=state['task_id'], **decision))
        return result['fields']

    def decide(self, states, stage, events):
        if not states:
            return []
        result = self.backend.decide(
            [observation(state, stage) for state in states], **DECISIONS[stage],
            mode=self.config['controller_mode'], max_new_tokens=self.config['controller_max_new_tokens'],
        )
        events.append(self.journal.event(
            'batch_completed', role='controller', stage=stage, gpu_id=self.rank,
            batch_size=len(states), duration_seconds=result['elapsed_seconds'],
            input_tokens=sum(result['input_tokens']), output_tokens=sum(result['output_tokens']),
        ))
        for index, state in enumerate(states):
            decision = {
                'stage': stage, 'choice': result['choices'][index], 'options': DECISIONS[stage]['options'],
                'probabilities': result['probabilities'][index], 'input_tokens': result['input_tokens'][index],
                'probability_status': 'Uncalibrated conditional option scores; null for generated JSON.',
            }
            state['decisions'].append(decision)
            events.append(self.journal.event('decision', task=state['task_id'], **decision))
        return result['choices']

    def spawn(self, states, role, events):
        if not states:
            return
        prompts = [observation(state, role, review_focus=role == 'review') for state in states]
        parents = [state['agent_id'] if role == 'review' else f"{state['task_id']}:review" for state in states]
        agents = [f"{state['task_id']}:{role}" for state in states]
        for state, prompt, agent, parent in zip(states, prompts, agents, parents):
            events.append(self.journal.event(
                'agent_spawned', state['task_id'], agent, parent, role, prompt=prompt, gpu_id=self.rank,
            ))
            events.append(self.journal.event('agent_started', state['task_id'], agent, parent, role, gpu_id=self.rank))
        result = self.backend.generate(prompts, SYSTEM[role], self.config['max_new_tokens'][role])
        events.append(self.journal.event(
            'batch_completed', role=role, gpu_id=self.rank, batch_size=len(states),
            duration_seconds=result['elapsed_seconds'], input_tokens=sum(result['input_tokens']),
            output_tokens=sum(result['output_tokens']),
        ))
        for index, (state, agent, parent) in enumerate(zip(states, agents, parents)):
            state[role] = result['texts'][index]
            if role == 'repair':
                state['revised_solution'] = code_from_response(state[role])
            usage = {'role': role, 'input_tokens': result['input_tokens'][index],
                     'output_tokens': result['output_tokens'][index], 'truncated': result['truncated'][index]}
            state['usage'].append(usage)
            events.append(self.journal.event(
                'agent_completed', state['task_id'], agent, parent, output=state[role], gpu_id=self.rank, **usage,
            ))

    def finish(self, states, events):
        policy = self.config['spawn_policy']
        if policy == 'single':
            return
        fields = self.review_fields(states, events)
        reviewers = states
        if policy == 'adaptive':
            choices = fields['review']['choices']
            reviewers = [state for state, choice in zip(states, choices) if choice == 'review']
        self.spawn(reviewers, 'review', events)
        repairers = reviewers
        if policy == 'adaptive':
            choices = self.decide(reviewers, 'repair', events)
            repairers = [state for state, choice in zip(reviewers, choices) if choice == 'repair']
        self.spawn(repairers, 'repair', events)
        choices = self.decide(repairers, 'select', events)
        for state, choice in zip(repairers, choices):
            if choice == 'revised':
                state['solution'] = state['revised_solution']
