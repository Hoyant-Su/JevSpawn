import argparse
import ast
import copy
import io
import json
from pathlib import Path

import torch

from baselines.agentprune.adapter import Dataset, configure
from baselines.agentprune.checkpoint import save
from baselines.agentprune.upstream import graph_kwargs, load_core, training_function


def main():
    parser = argparse.ArgumentParser()
    for name in ['source', 'config', 'prompts', 'tasks', 'labels', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    graph_class = load_core(args.source)
    configure(None, config['generation'], json.loads(args.prompts.read_text()))
    graph = graph_class(domain=config['domain'], llm_name='Qwen3.5-4B',
                        agent_names=config['agent_names'], decision_method=config['decision_method'],
                        optimized_spatial=config['optimized_spatial'],
                        optimized_temporal=config['optimized_temporal'],
                        **graph_kwargs(args.source)(config['mode'], len(config['agent_names'])))
    cloned = copy.deepcopy(graph)
    assert all(graph.nodes[k].llm is cloned.nodes[k].llm for k in graph.nodes)
    stream = io.BytesIO()
    torch.save(graph, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert list(restored.nodes) == list(graph.nodes)
    dataset = Dataset(args.tasks, args.labels, config['batch_size'])
    assert len(dataset) == config['task_count']
    for row in dataset.rows:
        for option in row['fields']['q0']['options']:
            assert dataset.postprocess_answer([option['id']]) == option['id']
    _, transformed = training_function(args.source, None)
    original = ast.parse((args.source / 'experiments/train_mmlu.py').read_text())
    assignments = {target.id: ast.dump(node.value) for node in ast.walk(original)
                   if isinstance(node, ast.Assign) for target in node.targets
                   if isinstance(target, ast.Name)}
    executed = {target.id: ast.dump(node.value) for node in ast.walk(ast.parse(transformed))
                if isinstance(node, ast.Assign) for target in node.targets
                if isinstance(target, ast.Name)}
    for name in ['optimizer', 'utility', 'single_loss', 'total_loss']:
        assert assignments[name] == executed[name]
    save(args.output, {'scope': 'CPU source, graph, transport and checkpoint qualification. No model inference.',
        'tasks': len(dataset), 'roles': [node.role for node in graph.nodes.values()],
        'spatial_mask_edges': int(graph.spatial_masks.sum()),
        'temporal_mask_edges': int(graph.temporal_masks.sum()),
        'original_optimizer_reward_and_loss_ast_preserved': True,
        'deepcopy_shares_transport': True, 'checkpoint_preserves_node_ids': True,
        'provided_option_letters_preserved': True})


if __name__ == '__main__':
    main()
