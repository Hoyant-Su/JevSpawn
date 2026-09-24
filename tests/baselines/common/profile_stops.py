import itertools
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, StopStringCriteria

from baselines.common.stops import matching_positions, StopCache


def main():
    words = [''.join(chars) for size in range(5) for chars in itertools.product('ab', repeat=size)]
    stops = words[1:]
    expected = StopStringCriteria._stop_string_get_matching_positions(words, list(range(len(words))), stops)
    assert matching_positions(words, list(range(len(words))), stops) == expected
    source = json.loads(Path('runs/common-react-aqua-development-001/session-0000/batches.json').read_text())
    unique = list(dict.fromkeys(tuple(stops) for batch in source for stops in batch['row_stops'] if stops))
    tokenizer = AutoTokenizer.from_pretrained('/inspire/hdd/global_public/public_models/Qwen/Qwen3.8-27B',
                                             local_files_only=True)
    fast = StopCache(tokenizer, 'cpu', 'indexed')
    results = []
    for stops in unique:
        started = time.perf_counter()
        original = StopStringCriteria(tokenizer, list(stops))
        old_seconds = time.perf_counter() - started
        started = time.perf_counter()
        candidate = fast.get(stops)
        new_seconds = time.perf_counter() - started
        equal = torch.equal(original.embedding_vec, candidate.embedding_vec)
        assert equal
        assert original.maximum_token_len == candidate.maximum_token_len
        assert original.max_valid_positions == candidate.max_valid_positions
        assert original.max_valid_end_lens == candidate.max_valid_end_lens
        assert torch.equal(original.target_lens, candidate.target_lens)
        results.append({'stops': stops, 'original_seconds': old_seconds, 'indexed_seconds': new_seconds,
                        'exact_table_parity': equal, 'table_shape': list(original.embedding_vec.shape)})
    Path('results/stop_table_qualification.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
