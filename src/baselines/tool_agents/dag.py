from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import re
import time

from baselines.tool_agents.tools import ActionError, execute


REFERENCE = re.compile(r'\$\{([A-Za-z0-9_-]+)\.([A-Za-z0-9_.]+)\}')


def references(value):
    if isinstance(value, dict):
        if set(value) == {'ref', 'path'}:
            return {value['ref']}
        return set().union(*(references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(references(item) for item in value))
    if isinstance(value, str):
        return {match[0] for match in REFERENCE.findall(value)}
    return set()


def lookup(results, identity, path):
    result = results[identity]
    for key in path:
        result = result[key]
    return result


def resolve(value, results):
    if isinstance(value, dict):
        if set(value) == {'ref', 'path'}:
            return lookup(results, value['ref'], value['path'])
        return {key: resolve(item, results) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item, results) for item in value]
    if isinstance(value, str):
        def replace(match):
            path = [int(part) if part.isdecimal() else part for part in match[2].split('.')]
            result = lookup(results, match[1], path)
            if type(result) not in (int, float):
                raise ActionError('Expression interpolation requires a numeric tool result.')
            return '(' + str(result) + ')'
        return REFERENCE.sub(replace, value)
    return value


def validate(nodes, inventory, maximum):
    if not isinstance(nodes, list) or not 0 <= len(nodes) <= maximum:
        raise ActionError('Plan must contain a bounded node list.')
    known = set()
    for node in nodes:
        if set(node) != {'id', 'tool', 'arguments', 'depends_on'}:
            raise ActionError('A plan node requires id, tool, arguments and depends_on.')
        identity = node['id']
        if not isinstance(identity, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identity) or identity in known:
            raise ActionError('Plan node IDs must be unique identifier strings.')
        if node['tool'] not in inventory or not isinstance(node['arguments'], dict):
            raise ActionError('Plan tool or arguments are invalid.')
        dependencies = node['depends_on']
        if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
            raise ActionError('Dependencies must be a list of node IDs.')
        if not set(dependencies) <= known or not references(node['arguments']) <= set(dependencies):
            raise ActionError('Dependencies must precede this node and include all result references.')
        known.add(identity)


def execute_plan(nodes, state, config, inventory, trace):
    validate(nodes, inventory, config['compiler_max_nodes'])
    pending, results = list(nodes), {}

    def call(node):
        arguments = resolve(node['arguments'], results)
        started = time.perf_counter()
        try:
            observation = execute(node['tool'], arguments, state, config['calculator'])
        except (ValueError, TypeError, KeyError, IndexError, SyntaxError, ZeroDivisionError, OverflowError) as error:
            return {'event': 'tool_failure', 'node': node['id'], 'tool': node['tool'],
                    'arguments': arguments, 'error': str(error),
                    'started': started, 'finished': time.perf_counter()}
        return {'event': 'observation', 'node': node['id'], 'tool': node['tool'],
                'arguments': arguments, 'observation': observation,
                'started': started, 'finished': time.perf_counter()}

    with ThreadPoolExecutor(max_workers=config['tool_concurrency']) as pool:
        running, failures = {}, []
        while pending or running:
            ready = [node for node in pending if set(node['depends_on']) <= results.keys()]
            for node in ready:
                trace.append({'event': 'tool_call', 'node': node['id'], 'tool': node['tool'], 'arguments': node['arguments']})
                running[pool.submit(call, node)] = node
                pending.remove(node)
            completed, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in completed:
                node = running.pop(future)
                event = future.result()
                trace.append(event)
                if event['event'] == 'tool_failure':
                    failures.append(event['error'])
                else:
                    results[node['id']] = event['observation']
            if failures:
                for future in running:
                    trace.append(future.result())
                raise ActionError('; '.join(failures))
    return results
