from collections import defaultdict

from methods.source_sets.inputs import partition


def partition_frontier(backend, indexed, references, fixed, prompts):
    membership = {reference['id']: set(reference['source_ids']) for reference in references}
    buckets = defaultdict(list)
    for identity, unit in indexed.items():
        owners = tuple(reference['id'] for reference in references if identity in membership[reference['id']])
        if owners:
            buckets[owners].append(unit)
    questions = {reference['id']: reference['question'] for reference in references}
    return [group for owners, units in buckets.items()
            for group in partition(backend, units, [(owner, questions[owner]) for owner in owners], fixed, prompts)]
