import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def soft_table_map(domains, factors, all_different, time_limit_seconds):
    """Maximize summed log table probabilities under exact finite constraints.

    >>> result = soft_table_map({'a': [1, 2], 'b': [1, 2]}, [
    ...     {'scope': ['a'], 'tuples': [[1], [2]], 'probabilities': [.4, .1]},
    ...     {'scope': ['b'], 'tuples': [[1], [2]], 'probabilities': [.3, .2]}],
    ...     [['a', 'b']], 10)
    >>> result['status'], result['assignment']
    ('optimal', {'a': 1, 'b': 2})
    >>> abs(result['log_probability'] - math.log(.4 * .2)) < 1e-10
    True
    >>> soft_table_map({'a': [1], 'b': [1]}, [], [['a', 'b']], 10)['status']
    'infeasible'
    """
    started = time.perf_counter()
    assert time_limit_seconds > 0
    assert all(values and len(values) == len(set(values)) for values in domains.values())
    indices = {(variable, value): index for index, (variable, value) in enumerate(
        (variable, value) for variable, values in domains.items() for value in values)}
    costs = [0.0] * len(indices)
    row_indices, column_indices, coefficients, lower, upper = [], [], [], [], []

    def constraint(terms, low, high):
        row = len(lower)
        for column, coefficient in terms:
            row_indices.append(row)
            column_indices.append(column)
            coefficients.append(coefficient)
        lower.append(low)
        upper.append(high)

    for variable, values in domains.items():
        constraint([(indices[variable, value], 1) for value in values], 1, 1)
    for group in all_different:
        assert len(group) == len(set(group)) and all(variable in domains for variable in group)
        for value in dict.fromkeys(value for variable in group for value in domains[variable]):
            constraint([(indices[variable, value], 1) for variable in group
                        if value in domains[variable]], 0, 1)
    for factor in factors:
        scope, tuples, probabilities = factor['scope'], factor['tuples'], factor['probabilities']
        assert len(scope) == len(set(scope)) and all(variable in domains for variable in scope)
        assert len(tuples) == len(probabilities)
        assert len({tuple(values) for values in tuples}) == len(tuples)
        assert all(len(values) == len(scope) and all(value in domains[variable]
                   for variable, value in zip(scope, values)) for values in tuples)
        assert all(math.isfinite(value) and 0 < value <= 1 for value in probabilities)
        offset = len(costs)
        costs.extend(-math.log(value) for value in probabilities)
        constraint([(offset + index, 1) for index in range(len(tuples))], 1, 1)
        for position, variable in enumerate(scope):
            for value in domains[variable]:
                terms = [(offset + index, 1) for index, values in enumerate(tuples)
                         if values[position] == value]
                constraint([*terms, (indices[variable, value], -1)], 0, 0)
    matrix = coo_matrix((coefficients, (row_indices, column_indices)),
                        shape=(len(lower), len(costs))).tocsc()
    build_seconds = time.perf_counter() - started
    result = milp(np.asarray(costs), integrality=np.ones(len(costs)),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, upper),
                  options={'time_limit': time_limit_seconds, 'mip_rel_gap': 0.0})
    assignment = None
    if result.x is not None:
        assignment = {variable: next(value for value in values if result.x[indices[variable, value]] > .5)
                      for variable, values in domains.items()}
    bound = getattr(result, 'mip_dual_bound', None)
    return {
        'status': {0: 'optimal', 1: 'limit', 2: 'infeasible', 3: 'unbounded', 4: 'solver_error'}[result.status],
        'solver_status': int(result.status), 'message': result.message, 'assignment': assignment,
        'log_probability': -float(result.fun) if result.fun is not None else None,
        'log_probability_upper_bound': -float(bound) if bound is not None else None,
        'relative_gap': getattr(result, 'mip_gap', None),
        'visited_nodes': getattr(result, 'mip_node_count', None),
        'build_seconds': build_seconds, 'total_seconds': time.perf_counter() - started,
        'time_limit_seconds': time_limit_seconds, 'binary_variables': len(costs),
        'constraint_rows': len(lower),
    }
