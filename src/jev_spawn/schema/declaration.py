from copy import deepcopy
from graphlib import TopologicalSorter
from heapq import heapify, heappop, heappush
from string import Template


def declaration_contract(schema, candidate_capacity):
    contract = deepcopy(schema)
    contract['properties']['fields']['items']['properties']['values']['maxItems'] = candidate_capacity
    return contract


def template_references(value, settings):
    if isinstance(value, str):
        template = Template(value)
        if not template.is_valid():
            raise ValueError('Invalid template expression: ' + value)
        return set(template.get_identifiers())
    if isinstance(value, dict):
        if set(value) == {settings['history_reference']}:
            return set()
        return set().union(*(template_references(child, settings) for child in value.values()))
    if isinstance(value, list):
        return set().union(*(template_references(child, settings) for child in value))
    return set()


def compile_declaration(declaration, settings):
    """Order template selectors before their parameters and retain referenced fields."""
    if not template_references(declaration['action'], settings):
        raise ValueError('Declare a variable action field using {name:enum(...)} or multiple native command lines.')
    fields = {field['id']: field for field in declaration['fields']}
    dependencies = {identity: template_references(field['values'], settings)
                    for identity, field in fields.items()}
    roots = template_references([declaration['action'], declaration['answer']], settings)
    missing = roots.union(*dependencies.values()) - fields.keys()
    if missing:
        raise ValueError('Undefined declaration fields: ' + ', '.join(sorted(missing)))
    predecessors = {identity: [] for identity in fields}
    for selector, parameters in dependencies.items():
        for parameter in parameters:
            predecessors[parameter].append(selector)
    graph = TopologicalSorter(predecessors)
    graph.prepare()
    positions = {identity: position for position, identity in enumerate(fields)}
    ready = [(positions[identity], identity) for identity in graph.get_ready()]
    heapify(ready)
    ordered = []
    while ready:
        _, identity = heappop(ready)
        ordered.append(identity)
        graph.done(identity)
        for child in graph.get_ready():
            heappush(ready, (positions[child], child))
    live = set(roots)
    for identity in ordered:
        if identity in live:
            live.update(dependencies[identity])
    compiled = deepcopy(declaration)
    compiled['fields'] = [deepcopy(fields[identity]) for identity in ordered if identity in live]
    return compiled
