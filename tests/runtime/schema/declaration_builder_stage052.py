import ast
from copy import deepcopy
import json
from math import ceil, floor
from string import Formatter
import time

import jsonschema


class DeclarationBuilder:
    """Compile compact native signatures into finite fields and fixed serializers."""

    def __init__(self, query, tools, answer_schema, service, task_id, budget, settings, prompts, execution, trace, *, feedback):
        self.query, self.tools, self.answer_schema = query, tools, answer_schema
        self.service, self.task_id, self.budget = service, task_id, budget
        self.settings, self.prompts, self.execution, self.trace = settings, prompts, execution, trace
        self.program = {'fields': [], 'action': {}, 'answer': None}
        self.bindings = {}
        self.feedback = feedback
        self.calls, self.tokens, self.decisions = [], 0, 0
        self.trace.update(calls=self.calls, generated_tokens=self.tokens, finite_decisions=self.decisions,
                          named_bindings=self.bindings, declaration=self.program)

    def state(self, detail):
        return json.dumps({'construction_limits': self.settings,
            'generation_tokens_remaining': self.budget['max_new_tokens'] - self.tokens,
            'public_tools': self.tools, 'answer_schema': self.answer_schema,
            'partial_declaration': self.program, 'named_bindings': self.bindings, 'feedback': self.feedback,
            'component': detail}, **self.execution['serialization'])

    def choose(self, question, options, detail):
        assert options, 'No supported construction option: ' + str(detail)
        self.decisions += 1
        assert self.decisions <= self.settings['max_construction_decisions'], 'Declaration decision budget exhausted.'
        request = {'id': self.settings['decision_id'].format(index=self.decisions),
            'context': self.query, 'state': self.state(detail), 'question': self.prompts['choose'].format(question=question),
            'options': [{'id': str(index), 'description': json.dumps(option, **self.execution['serialization'])}
                        for index, option in enumerate(options)]}
        started = time.perf_counter()
        decision, = self.service.decide([request], task_id=self.task_id)
        self.calls.append({'kind': 'finite', 'requests': [request], 'outputs': [decision],
                           'elapsed_seconds': time.perf_counter() - started})
        self.trace['finite_decisions'] = self.decisions
        return options[int(decision['choice'])]

    def signature(self, schema, path):
        component = {'destination': path, 'schema': schema, 'phase': 'action',
                     'existing_named_domains': self.bindings, 'constant_string': False}
        cap = self.budget['max_new_tokens'] - self.tokens
        assert cap > 0, 'Declaration generation budget exhausted.'
        system = self.prompts[self.settings['signature_prompt']].format(
            component=json.dumps(component, **self.execution['serialization']),
            contract=json.dumps(self.settings, **self.execution['serialization']))
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content':
            self.prompts['signature_user'].format(context=self.query, state=self.state(component),
                control_feedback=json.dumps({key: value for key, value in self.feedback.items()
                    if key != 'execution'}, **self.execution['serialization']))}]
        started = time.perf_counter()
        output, = self.service.complete_batch([messages], cap, self.budget['temperature'], None,
                                              task_id=self.task_id, return_tokens=True)
        self.tokens += len(output['token_ids'])
        self.trace['generated_tokens'] = self.tokens
        self.calls.append({'kind': 'signature', 'requests': [messages], 'outputs': [output],
                          'max_new_tokens': cap,
                          'elapsed_seconds': time.perf_counter() - started})
        if output['finish_reason'] == 'length':
            raise ValueError(f'Declaration generation exhausted its remaining {cap}-token budget; '
                             'the incomplete signature was not compiled.')
        if output['finish_reason'] == 'timeout':
            raise TimeoutError('Declaration generation exceeded the task deadline.')
        return self.compile_signature(output['text'], path)

    def domain(self, expression):
        return self.domain_node(ast.parse(expression, mode='eval').body)

    def domain_node(self, node):
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords, 'Unsupported domain expression.'
        operator = node.func.id
        if operator == 'enum':
            values = [ast.literal_eval(argument) for argument in node.args]
            assert all(type(value) in (str, int, float, bool, type(None)) for value in values), 'An enum requires scalar literals.'
            return self.finite_domain(values)
        if operator == 'range':
            assert len(node.args) == 2, 'range requires start and stop.'
            start, stop = [ast.literal_eval(argument) for argument in node.args]
            assert type(start) is int and type(stop) is int, 'range endpoints must be integers.'
            assert self.settings['min_candidates'] <= stop - start <= self.settings['max_candidates'], 'Range exceeds finite domain capacity.'
            return self.finite_domain(list(range(start, stop)))
        if operator == 'repeat':
            assert len(node.args) == 2, 'repeat requires count and domain.'
            count = ast.literal_eval(node.args[0])
            assert type(count) is int and self.settings['min_repeat'] <= count <= self.settings['max_fields'], 'Invalid repeat count.'
            return {'repeat': count, 'domain': self.domain_node(node.args[1])}
        raise ValueError('Unsupported domain operator: ' + operator)

    def finite_domain(self, values):
        assert self.settings['min_candidates'] <= len(values) <= self.settings['max_candidates'], 'Finite domain exceeds configured capacity.'
        keys = [json.dumps(value, sort_keys=True) for value in values]
        assert len(set(keys)) == len(keys), 'Finite domain contains duplicate members.'
        return {'values': values}

    def escape(self, value):
        return value.replace('$', '$$') if isinstance(value, str) else value

    def reference(self, identity):
        return self.settings['field_reference'].format(identity=identity)

    def field(self, role, values):
        assert len(self.program['fields']) < self.settings['max_fields'], 'Declaration field capacity exhausted.'
        assert self.settings['min_candidates'] <= len(values) <= self.settings['max_candidates'], 'Field domain capacity exhausted.'
        identity = self.settings['field_id'].format(index=len(self.program['fields']))
        self.program['fields'].append({'id': identity, 'question': role, 'values': values})
        return self.reference(identity)

    def allocate(self, role, domain):
        if 'repeat' in domain:
            return ''.join(self.allocate(self.settings['indexed_role'].format(role=role, index=index), domain['domain'])
                           for index in range(domain['repeat']))
        return self.field(role, [self.escape(value) for value in domain['values']])

    def bind(self, name, specification):
        assert name.isidentifier(), 'A slot role must be an identifier: ' + name
        if name in self.bindings:
            binding = self.bindings[name]
            if specification:
                assert json.dumps(self.domain(specification), sort_keys=True) == json.dumps(binding['domain'], sort_keys=True), 'Reused role has a different domain: ' + name
            return binding['template']
        assert specification, 'Undefined signature role: ' + name
        domain = self.domain(specification)
        template = self.allocate(name, domain)
        self.bindings[name] = {'domain': domain, 'template': template}
        return template

    def compile_signature(self, text, path):
        variants = text.splitlines() or ['']
        assert len(variants) <= self.settings['max_variants'], 'Native signature variant capacity exhausted.'
        compiled = []
        for variant in variants:
            pieces = []
            for literal, name, specification, conversion in Formatter().parse(variant):
                assert conversion is None, 'Signature conversions are unsupported.'
                pieces.append(self.escape(literal))
                if name is not None:
                    pieces.append(self.bind(name, specification))
            compiled.append(''.join(pieces))
        assert len(set(compiled)) == len(compiled), 'Duplicate native signature variants.'
        self.calls.append({'kind': 'compiled_signature', 'path': path, 'native_variants': variants,
                          'compiled_variants': compiled, 'named_bindings': deepcopy(self.bindings)})
        if len(compiled) == 1:
            return compiled[0]
        role = self.settings['variant_role'].format(path=json.dumps(path))
        return self.field(role, compiled)

    def public_domain(self, schema):
        if 'enum' in schema:
            return self.finite_domain(schema['enum'])
        if 'const' in schema:
            return self.finite_domain([schema['const']])
        if schema['type'] == 'boolean':
            return self.finite_domain([False, True])
        if schema['type'] == 'null':
            return self.finite_domain([None])
        if schema['type'] == 'integer' and 'minimum' in schema and 'maximum' in schema:
            start, stop = ceil(schema['minimum']), floor(schema['maximum']) + 1
            assert stop - start <= self.settings['max_candidates'], 'Public integer range exceeds capacity.'
            values = [value for value in range(start, stop) if jsonschema.Draft202012Validator(schema).is_valid(value)]
            return self.finite_domain(values)
        raise ValueError('No explicit finite domain for public scalar schema: ' + json.dumps(schema))


    def node(self, schema, path, depth):
        assert depth <= self.settings['max_tree_depth'], 'Declaration type tree exceeds configured depth.'
        if 'oneOf' in schema or 'anyOf' in schema:
            alternatives = schema['oneOf'] if 'oneOf' in schema else schema['anyOf']
            schema = self.choose(self.prompts['schema_variant'], alternatives, {'path': path})
        if isinstance(schema['type'], list):
            kind = self.choose(self.prompts['value_type'], schema['type'], {'path': path})
            schema = {**schema, 'type': kind}
        kind = schema['type']
        if kind == 'object':
            values = {}
            for key, child in schema['properties'].items():
                present = key in schema.get('required', []) or self.choose(self.prompts['optional_key'],
                    [False, True], {'path': [*path, key], 'schema': child})
                if present:
                    values[key] = self.node(child, [*path, key], depth + 1)
            return values
        if kind == 'array':
            lower = schema.get('minItems', self.settings['min_array_items'])
            upper = min(schema.get('maxItems', self.settings['max_array_items']), self.settings['max_array_items'])
            count = self.choose(self.prompts['array_count'], list(range(lower, upper + 1)), {'path': path, 'schema': schema})
            return [self.node(schema['items'], [*path, index], depth + 1) for index in range(count)]
        if kind == 'string' and not ('enum' in schema or 'const' in schema):
            return self.signature(schema, path)
        return self.allocate(json.dumps(path), self.public_domain(schema))

    def build_action(self):
        started = time.perf_counter()
        name = self.choose(self.prompts['tool'], [name for name in self.tools
                            if name != self.settings['submission_tool']], {})
        self.action_schema = self.tools[name]['input_schema']
        self.program['action'] = {'tool': name, 'arguments': {}}
        self.program['action']['arguments'] = self.node(self.action_schema,
            self.settings['action_root'], self.settings['initial_depth'])
        self.trace['action_construction_seconds'] = time.perf_counter() - started
        return self.program
