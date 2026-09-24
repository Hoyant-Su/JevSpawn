def _tables(domains, factors):
    tables = [(tuple(factor['scope']), tuple(tuple(row) for row in factor['allowed']))
              for factor in factors]
    for scope, rows in tables:
        assert len(scope) == len(set(scope))
        assert all(variable in domains for variable in scope)
        assert all(len(row) == len(scope) for row in rows)
    return tables


def _propagate(state, tables):
    changes = []
    if any(not values for values in state.values()):
        return False, changes
    changed = True
    while changed:
        changed = False
        for factor_index, (scope, rows) in enumerate(tables):
            supported = [row for row in rows if all(
                value in state[variable] for variable, value in zip(scope, row))]
            if not supported:
                return False, changes
            for index, variable in enumerate(scope):
                values = [value for value in state[variable]
                          if any(row[index] == value for row in supported)]
                if len(values) < len(state[variable]):
                    changes.append({'variable': variable, 'before': list(state[variable]),
                                    'after': list(values), 'factor_index': factor_index})
                    state[variable] = values
                    changed = True
    return True, changes


def propagate(domains, factors):
    """Apply exact generalized arc consistency without search or input mutation.

    Changes list strict domain revisions in execution order. Factor indices refer
    to the supplied list. A contradiction stops propagation immediately.

    >>> domains = {'x': [1, 0], 'y': [0]}
    >>> result = propagate(domains, [{'scope': ['x', 'y'], 'allowed': [(0, 0), (1, 1)]}])
    >>> result['domains'], domains
    ({'x': [0], 'y': [0]}, {'x': [1, 0], 'y': [0]})
    >>> result['changes']
    [{'variable': 'x', 'before': [1, 0], 'after': [0], 'factor_index': 0}]
    >>> chain = [{'scope': pair, 'allowed': [(0, 0), (1, 1)]}
    ...          for pair in [('x', 'y'), ('y', 'z')]]
    >>> result = propagate({'x': [0, 1], 'y': [0, 1], 'z': [1]}, chain)
    >>> result['domains'], [change['factor_index'] for change in result['changes']]
    ({'x': [1], 'y': [1], 'z': [1]}, [1, 0])
    >>> propagate({}, [{'scope': [], 'allowed': [[]]}])['status']
    'consistent'
    >>> propagate({}, [{'scope': [], 'allowed': []}])['status']
    'unsat'
    """
    state = {key: list(values) for key, values in domains.items()}
    consistent, changes = _propagate(state, _tables(domains, factors))
    return {'status': 'consistent' if consistent else 'unsat', 'domains': state,
            'domain_updates': len(changes), 'values_removed': sum(
                len(change['before']) - len(change['after']) for change in changes),
            'changes': changes}


def solve(domains, factors, max_nodes):
    """Solve finite table constraints in the supplied variable and value order.

    Each visited search node includes propagation, including the root. Domain
    updates count strict reductions made by propagation across all visited nodes.
    The node budget does not limit the number of propagation revisions.

    >>> equality = {'scope': ['x', 'y'], 'allowed': [(0, 0), (1, 1)]}
    >>> solve({'x': [1, 0], 'y': [0, 1]}, [equality], 2)['assignment']
    {'x': 1, 'y': 1}
    >>> triangle = [{'scope': pair, 'allowed': [(0, 1), (1, 0)]}
    ...             for pair in [('a', 'b'), ('b', 'c'), ('a', 'c')]]
    >>> solve({'a': [0, 1], 'b': [0, 1], 'c': [0, 1]}, triangle, 3)['status']
    'unsat'
    >>> parity = {'scope': ['x', 'y', 'z'],
    ...           'allowed': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]}
    >>> solve({'x': [0, 1], 'y': [1], 'z': [1]}, [parity], 1)['assignment']
    {'x': 0, 'y': 1, 'z': 1}
    >>> solve({'x': [0, 1], 'y': [0, 1]}, [equality], 1)['status']
    'budget'
    """
    assert isinstance(max_nodes, int) and max_nodes >= 0
    variables = list(domains)
    tables = _tables(domains, factors)
    visited = updates = removed = 0

    def search(state):
        nonlocal visited, updates, removed
        if visited == max_nodes:
            return 'budget', None
        visited += 1
        consistent, changes = _propagate(state, tables)
        updates += len(changes)
        removed += sum(len(change['before']) - len(change['after']) for change in changes)
        if not consistent:
            return 'unsat', None
        remaining = [key for key in variables if len(state[key]) > 1]
        if not remaining:
            return 'sat', {key: state[key][0] for key in variables}
        variable = remaining[0]
        for value in state[variable]:
            child = {key: list(values) for key, values in state.items()}
            child[variable] = [value]
            status, assignment = search(child)
            if status != 'unsat':
                return status, assignment
        return 'unsat', None

    status, assignment = search({key: list(values) for key, values in domains.items()})
    return {'status': status, 'assignment': assignment, 'visited_nodes': visited,
            'domain_updates': updates, 'values_removed': removed}
