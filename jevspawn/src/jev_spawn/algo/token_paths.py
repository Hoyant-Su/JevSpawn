def compile_token_paths(paths):
    """Compile native token paths; forced runs still require causal model extension."""
    paths = tuple(tuple(path) for path in paths)
    if not paths or any(not path for path in paths):
        raise ValueError('Native token paths must be nonempty.')
    token_ids = sorted({token for path in paths for token in path})
    if any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError('Native token IDs must be nonnegative integers.')
    token_indices = {token: index for index, token in enumerate(token_ids)}
    edges, terminal = [{}], [False]
    root = 0
    for path in paths:
        state = root
        for token in path:
            if token not in edges[state]:
                edges[state][token] = len(edges)
                edges.append({})
                terminal.append(False)
            state = edges[state][token]
        terminal[state] = True
    if any(done and children for done, children in zip(terminal, edges, strict=True)):
        raise ValueError('A terminal path prefixes another path; supply explicit terminator tokens.')
    width = max(map(len, edges))
    edge_token_indices, edge_next_states, edge_valid = [], [], []
    forced, forced_next_states = [], []
    for state, children in enumerate(edges):
        ordered = sorted(children.items())
        padding = width - len(ordered)
        # Padding is inactive under edge_valid or forced_lengths, never an emitted token.
        edge_token_indices.append([token_indices[token] for token, _ in ordered] + [root] * padding)
        edge_next_states.append([target for _, target in ordered] + [root] * padding)
        edge_valid.append([True] * len(ordered) + [False] * padding)
        run, target = [], state
        while len(edges[target]) == 1:
            token, target = next(iter(edges[target].items()))
            run.append(token_indices[token])
        forced.append(run)
        forced_next_states.append(target)
    forced_width = max(map(len, forced))
    return {'token_ids': token_ids, 'edge_token_indices': edge_token_indices,
            'edge_next_states': edge_next_states, 'edge_valid': edge_valid,
            'terminal': terminal, 'root': root,
            'forced_token_indices': [run + [root] * (forced_width - len(run)) for run in forced],
            'forced_lengths': list(map(len, forced)), 'forced_next_states': forced_next_states}
