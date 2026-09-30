from copy import deepcopy
from graphlib import TopologicalSorter
import json
import time

import torch

from jev_spawn.runtime.branches import spawn
from jev_spawn.runtime.query_execution import compiled_string
from jev_spawn.schema.declaration import template_references


def substitute_bound(value, bindings):
    """Materialize known bindings and leave unresolved template references intact."""
    if isinstance(value, dict):
        return {key: substitute_bound(child, bindings) for key, child in value.items()}
    if isinstance(value, list):
        return [substitute_bound(child, bindings) for child in value]
    if isinstance(value, str):
        template, names, reference = compiled_string(value)
        resolved = {name: substitute_bound(bindings[name], bindings) for name in names if name in bindings}
        if reference and resolved:
            name, = names
            return resolved[name]
        return template.safe_substitute(resolved)
    return deepcopy(value)


def prepare_actions(original, execution, fields):
    domains, requests, evidence = original(execution, fields)
    for request in requests:
        identity = request['field_id']
        for option in request['options']:
            value = domains[identity][option['id']]
            action = substitute_bound(execution.action_template, {**execution.bound_fields, identity: value})
            option['description'] = json.dumps({'value': value, 'resulting_action': action},
                                                **execution.settings['serialization'])
    return domains, requests, evidence


def merge_extensions(beam, active, selected, choices, extension_scores, prefix_scores, width):
    """Completed actions and conditional extensions compete in the same beam."""
    inactive = [index for index in range(len(beam)) if index not in active]
    carried = torch.tensor(inactive, dtype=torch.long, device=prefix_scores.device)
    candidates = [(index, None) for index in inactive]
    candidates.extend((active[index], choice) for index, choice in
                      zip(selected, choices, strict=True))
    scores = torch.cat((prefix_scores.index_select(0, carried), extension_scores))
    order = scores.argsort(descending=True, stable=True)[:width]
    return [candidates[index] for index in order.tolist()], scores[order]


def field_layers(declaration, settings):
    fields = declaration['fields']
    predecessors = {field['id']: set() for field in fields}
    for field in fields:
        for dependency in template_references(field['values'], settings):
            predecessors[dependency].add(field['id'])
    graph = TopologicalSorter(predecessors)
    graph.prepare()
    layers = []
    while graph.is_active():
        ready = graph.get_ready()
        layers.append([field for field in fields if field['id'] in ready])
        graph.done(*ready)
    return layers


def dispatch(branches, prepared, workers, identities, timing):
    proposals, selections = [], {}
    for _, owner, execution, log_probability in sorted(prepared, key=lambda item: item[0]):
        identity = next(identities)
        values = dict(execution.bound_fields)
        selections[identity] = (values, log_probability, owner)
        proposals.append({'id': identity, 'parent': owner, 'values': values,
                          'action': branches[owner].declaration['action']})
    dispatch_started = time.perf_counter()
    children = spawn(branches, proposals, workers)
    dispatch_seconds = time.perf_counter() - dispatch_started
    for identity, child in children.items():
        values, log_probability, owner = selections[identity]
        child.execution.trace.append({'parent_id': owner, 'selected_values': values,
            'layer_factored_log_probability': log_probability,
            'shared_batch_timing': {**timing, 'dispatch_seconds': dispatch_seconds,
                                   'executed_actions': len(children)}})
    return children
