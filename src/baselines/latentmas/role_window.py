from fractions import Fraction


def role_budgets(demands, capacity, policy):
    assert policy['allocation'] == 'weighted_capped'
    weights = [Fraction(str(value)) for value in policy['weights']]
    order = policy['order']
    assert len(weights) == len(demands) and set(order) == set(range(len(demands)))
    assert len(order) == len(demands) and all(weight > 0 for weight in weights)
    assert capacity > 0 and all(demand > 0 for demand in demands)
    if sum(demands) <= capacity:
        return list(demands)
    result = [0] * len(demands)
    active, remaining = list(order), capacity
    while active:
        total = sum(weights[index] for index in active)
        quotas = {index: remaining * weights[index] / total for index in active}
        saturated = [index for index in active if demands[index] <= quotas[index]]
        if saturated:
            for index in saturated:
                result[index] = demands[index]
                remaining -= demands[index]
            active = [index for index in active if index not in saturated]
        else:
            for index in active:
                result[index] = int(quotas[index])
            remaining -= sum(result[index] for index in active)
            ranked = sorted(active, key=lambda index: quotas[index] - result[index], reverse=True)
            for index in ranked[:remaining]:
                result[index] += 1
            active = []
    assert sum(result) == capacity and all(0 < cap <= demand for cap, demand in zip(result, demands, strict=True))
    return result
