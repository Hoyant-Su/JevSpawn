from collections import Counter

from methods.source_sets.analysis import validate_hierarchy


def hierarchy_statistics(result, calls):
    requests = result['retrieval_requests']
    active = {request['id']: request['source_ids'] for request in requests}
    assert len(active) == len(requests)
    assert list(active) == [reference['id'] for reference in result['references']]
    assert all(len(values) == len(set(values)) for values in active.values())
    sources = {unit['id'] for unit in result['sources']}
    assert set().union(*map(set, active.values())) == sources
    statistics = validate_hierarchy(result, calls, active)
    statistics.update(request_source_pairs=sum(map(len, active.values())),
                      dense_request_source_pairs=len(sources) * len(requests))
    return statistics


def validate_retrieval(result, trace):
    searches = [entry for entry in trace['operations'] if entry['operation'] == 'search']
    requests = result['retrieval_requests']
    assert len(searches) == len(requests)
    for request, search in zip(requests, searches):
        assert request['question'] == search['query']
        assert Counter(request['source_ids']) == Counter(hit['id'] for hit in search['results'])
