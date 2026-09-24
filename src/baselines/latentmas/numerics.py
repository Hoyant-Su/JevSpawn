import torch


def difference(left, right):
    assert left.shape == right.shape
    delta = left.float() - right.float()
    norm = float(right.float().norm())
    return {'shape': list(left.shape), 'dtype': str(left.dtype),
            'max_absolute': float(delta.abs().max()), 'delta_l2': float(delta.norm()),
            'reference_l2': norm, 'relative_l2': float(delta.norm()) / norm if norm else None,
            'exact_fraction': float((left == right).float().mean())}


def compare(batched, single, row, padded):
    states = []
    for step, (left, right) in enumerate(zip(batched['states'], single['states'])):
        left_mask, right_mask = left['mask'][row].bool(), right['mask'][0].bool()
        left_query, right_query = left['query_mask'][row].bool(), right['query_mask'][0].bool()
        assert int(left_mask.sum()) == int(right_mask.sum())
        assert torch.equal(left['positions'][row][left_query], right['positions'][0][right_query])
        if padded:
            assert torch.equal(left['mask'][row], right['mask'][0])
            assert torch.equal(left['positions'][row], right['positions'][0])
        caches = []
        for index, (batch_layer, layer) in enumerate(zip(left['layers'], right['layers'])):
            assert left['cache_classes'][index] == right['cache_classes'][index]
            tensors = {}
            for name, value in layer.items():
                batch_value, value = batch_layer[name][row], value[0]
                if name in ['keys', 'values']:
                    batch_value, value = batch_value[:, left_mask], value[:, right_mask]
                tensors[name] = difference(batch_value, value)
            caches.append({'layer': index, 'cache_class': left['cache_classes'][index], 'tensors': tensors})
        states.append({'step': step, 'phase': 'prefill' if step == 0 else 'latent_transition',
                       'valid_history_tokens': int(right_mask.sum()),
                       'batch_physical_cache_length': left['physical_cache_length'],
                       'single_physical_cache_length': right['physical_cache_length'],
                       'valid_positions_equal': True, 'full_masks_equal': torch.equal(left['mask'][row], right['mask'][0]),
                       'hidden': difference(left['hidden'][row], right['hidden'][0]),
                       'input_last': difference(left['input_last'][row], right['input_last'][0]),
                       'cache': caches})
    alignments = [{'transition': step + 1, **{name: difference(left[name][row], right[name][0])
                    for name in ['hidden_input', 'pre_aligned', 'aligned_output']}}
                  for step, (left, right) in enumerate(zip(batched['alignments'], single['alignments']))]
    initial = states[0]['hidden']['delta_l2']
    return {'states': states, 'alignments': alignments,
            'hidden_delta_l2_amplification_from_prefill': [s['hidden']['delta_l2'] / initial
                                                         if initial else None for s in states]}
