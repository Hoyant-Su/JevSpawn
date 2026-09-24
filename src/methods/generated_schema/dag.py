from collections import defaultdict
import json

import jsonschema


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('Duplicate JSON keys')
    return result


def validate_plan(text, schema, max_depth):
    plan = json.loads(text, object_pairs_hook=unique_object)
    jsonschema.validate(plan, schema)
    nodes = {node['id']: node for node in plan['nodes']}
    if len(nodes) != len(plan['nodes']):
        raise ValueError('Duplicate node IDs')
    for node in nodes.values():
        options = node['options']
        if len({option['id'] for option in options}) != len(options):
            raise ValueError('Duplicate option IDs')
        if len({option['description'] for option in options}) != len(options):
            raise ValueError('Duplicate option descriptions')
        if not set(node['parents']) <= nodes.keys() or node['id'] in node['parents']:
            raise ValueError('Invalid parent reference')
        condition = node.get('activate_if')
        if condition is not None:
            if condition['parent'] not in node['parents']:
                raise ValueError('Activation condition must reference a direct parent')
            if condition['equals'] not in {option['id'] for option in nodes[condition['parent']]['options']}:
                raise ValueError('Activation condition references an unknown option')
    depths = {}
    while len(depths) < len(nodes):
        ready = [node for node in nodes.values() if node['id'] not in depths and set(node['parents']) <= depths.keys()]
        if not ready:
            raise ValueError('Decision graph contains a cycle')
        for node in ready:
            depths[node['id']] = 1 + max((depths[parent] for parent in node['parents']), default=0)
    if max(depths.values()) > max_depth:
        raise ValueError('Decision graph exceeds the declared depth budget')
    return plan, depths


def original_input(row):
    return {'problem': row['state'], 'target_fields': row['fields']}


def eligible_groups(root):
    groups = defaultdict(list)
    completed = root['outcomes']
    for node in root['plan']['nodes']:
        if node['id'] in completed or not set(node['parents']) <= completed.keys():
            continue
        skipped_parent = any(completed[parent]['status'] == 'skipped' for parent in node['parents'])
        condition = node.get('activate_if')
        condition_false = condition is not None and not skipped_parent and completed[condition['parent']]['choice'] != condition['equals']
        if skipped_parent or condition_false:
            completed[node['id']] = {'status': 'skipped', 'reason': 'parent_skipped' if skipped_parent else 'condition_false'}
        else:
            groups[tuple(sorted(node['parents']))].append(node)
    return list(groups.items())


def outcome_context(root, node_id):
    node = next(node for node in root['plan']['nodes'] if node['id'] == node_id)
    result = root['outcomes'][node_id]
    return {'node': node, 'result': result}


def prepare_unit(root, parents, nodes):
    state = json.dumps({**original_input(root['task']),
                        'parent_results': [outcome_context(root, parent) for parent in parents]}, ensure_ascii=False)
    fields = {f'f{index}': {'question': node['question'],
                           'options': [{'id': f'o{slot}', 'description': option['description']}
                                       for slot, option in enumerate(node['options'])]}
              for index, node in enumerate(nodes)}
    return {'root': root, 'nodes': nodes, 'parents': list(parents), 'state': state, 'fields': fields}


def compatible_batches(units, batch_size):
    buckets = defaultdict(list)
    for unit in units:
        signature = tuple(len(field['options']) for field in unit['fields'].values())
        buckets[signature].append(unit)
    return [bucket[start:start + batch_size] for bucket in buckets.values()
            for start in range(0, len(bucket), batch_size)]
