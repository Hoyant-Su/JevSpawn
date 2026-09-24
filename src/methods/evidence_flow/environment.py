from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import re
from types import MappingProxyType


DOCUMENT_FIELDS = ('title', 'author', 'source', 'published_at', 'category', 'url')


class InvalidSearchQuery(ValueError):
    pass


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def group_by_document(units):
    groups = {}
    for unit in units:
        identifier = unit['doc_id']
        if identifier not in groups:
            groups[identifier] = {'id': identifier, **{key: unit[key] for key in DOCUMENT_FIELDS},
                                  'source_units': []}
        groups[identifier]['source_units'].append({key: unit[key] for key in ('id', 'start', 'end', 'text')})
    return list(groups.values())


class EvidenceEnvironment:
    def __init__(self, corpus_path, source_units_path, *, bm25_k1, bm25_b, token_pattern,
                 metadata_fields, search_limit, read_limit):
        documents, units = read_jsonl(corpus_path), read_jsonl(source_units_path)
        assert len({row['id'] for row in documents}) == len(documents)
        assert len({row['id'] for row in units}) == len(units)
        assert bm25_k1 > 0 and 0 <= bm25_b <= 1
        assert 1 <= search_limit <= len(units) and read_limit >= 1
        assert set(metadata_fields) <= set(DOCUMENT_FIELDS)
        self.documents = MappingProxyType({row['id']: MappingProxyType(row) for row in documents})
        self.units = MappingProxyType({row['id']: MappingProxyType(row) for row in units})
        self.ids = tuple(row['id'] for row in units)
        for row in units:
            assert row['text'] == self.documents[row['doc_id']]['body'][row['start']:row['end']]
        self.pattern = re.compile(token_pattern)
        self.config = MappingProxyType({'bm25_k1': bm25_k1, 'bm25_b': bm25_b,
                                        'token_pattern': token_pattern, 'metadata_fields': tuple(metadata_fields),
                                        'search_limit': search_limit, 'read_limit': read_limit})
        postings, lengths = defaultdict(list), []
        for index, unit in enumerate(units):
            document = self.documents[unit['doc_id']]
            metadata = [document[key] for key in metadata_fields if document[key] is not None]
            terms = Counter(self.pattern.findall(('\n'.join([unit['text'], *metadata])).lower()))
            lengths.append(sum(terms.values()))
            for term, count in terms.items():
                postings[term].append((index, count))
        self.lengths = tuple(lengths)
        self.average_length = sum(lengths) / len(lengths)
        assert self.average_length > 0
        self.postings = MappingProxyType({term: tuple(matches) for term, matches in postings.items()})

    def episode(self, task_id):
        return EvidenceEpisode(self, task_id)

    def metadata(self, unit):
        document = self.documents[unit['doc_id']]
        return {key: document[key] for key in DOCUMENT_FIELDS}


class EvidenceEpisode:
    def __init__(self, environment, task_id):
        self.environment, self.task_id = environment, task_id
        self._trace = []

    @property
    def trace(self):
        return {'task_id': self.task_id, 'operations': deepcopy(self._trace)}

    def search(self, query, k):
        environment = self.environment
        assert 1 <= k <= environment.config['search_limit']
        terms = Counter(environment.pattern.findall(query.lower()))
        if not terms:
            raise InvalidSearchQuery('Search query must contain a lexical token.')
        scores = [0.0] * len(environment.ids)
        k1, b = environment.config['bm25_k1'], environment.config['bm25_b']
        for term, query_count in terms.items():
            matches = environment.postings.get(term, ())
            idf = math.log(1 + (len(scores) - len(matches) + 0.5) / (len(matches) + 0.5))
            for index, count in matches:
                denominator = count + k1 * (1 - b + b * environment.lengths[index] / environment.average_length)
                scores[index] += query_count * idf * count * (k1 + 1) / denominator
        ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], environment.ids[index]))[:k]
        results = []
        for index in ordered:
            unit = environment.units[environment.ids[index]]
            results.append({'id': unit['id'], 'doc_id': unit['doc_id'],
                            **environment.metadata(unit), 'score': scores[index]})
        self._trace.append({'operation': 'search', 'query': query, 'k': k, 'results': deepcopy(results)})
        return results

    def read(self, ids):
        environment = self.environment
        assert 1 <= len(ids) <= environment.config['read_limit']
        results = [{**environment.units[identifier], **environment.metadata(environment.units[identifier])}
                   for identifier in ids]
        self._trace.append({'operation': 'read', 'ids': list(ids), 'results': deepcopy(results)})
        return results
