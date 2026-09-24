import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from baselines.common.agentprune import Provider as OriginalProvider
from baselines.common.config import SharedConfig
from baselines.common.errors import InvalidOutputError
from environments.llfbench import LLFEnvironment
from agentprune_candidate import Provider, solve


method = json.loads(Path('configs/agentprune_protocol_repair_method_20260924.json').read_text())
sys.path[:0] = method['python_paths']
source = 'runs/native-baselines-formal-20260923/agentprune/llfbench_gridworld/session-0000/batches.json'
batch = json.loads(Path(source).read_text())[26]
output = batch['texts'][4]
settings = SharedConfig.load('configs/shared_config_tp4_v2.yaml').method_settings(method['settings'])
prompts = json.loads(Path(method['settings']['qualification_templates']).read_text())
old_prompts = json.loads(Path('configs/baselines/common/schema/agentprune.json').read_text())
tasks = [json.loads(line) for line in Path('data/native_formal_20260923/llfbench_gridworld/tasks.jsonl').read_text().splitlines()]
task = next(task for task in tasks if task['task_id'] == batch['task_ids'][4])
environment = LLFEnvironment(task, json.loads(Path('configs/environments/session.json').read_text()),
    Path('results/validation/agentprune_protocol_repair_20260924/replay_tools'),
    deadline=lambda: settings['sample_timeout_seconds'],
    configuration='configs/evaluation/native_context/llfbench_gridworld.json')
node = SimpleNamespace(id='recorded-regression', role='Historian')
complete = lambda *args: [output]
try:
    asyncio.run(OriginalProvider(node, False, environment, complete, settings, old_prompts, []).agen(batch['messages'][4]))
except InvalidOutputError as error:
    original_error = str(error)
else:
    raise AssertionError('Original fatal mismatch was not reproduced.')
calls = []
proposal = asyncio.run(Provider(node, False, environment, complete, settings, prompts, calls).agen(batch['messages'][4]))
assert json.loads(proposal) == json.loads(output.removeprefix('ToolCall:'))['arguments']
assert environment.answer is None and not environment.done and environment.actions == []
assert 'observation' not in calls[0] and 'done' not in calls[0]
result = solve(task, environment, complete, settings, old_prompts)
assert result['termination'] == 'graph_completed'
assert len(result['nodes']) == 5 and all(node['outputs'] == [proposal] for node in result['nodes'])
assert len(environment.actions) == 1 and environment.actions[0]['tool'] == 'finish'
assert result['decision_node_outputs'] == [proposal]
report = {'scope': 'CPU interface regression only, replaying an authentic saved model output; no model generation or benchmark score',
          'source': source, 'batch': 26, 'row': 4, 'task_id': task['task_id'], 'original_error': original_error,
          'proposal': json.loads(proposal), 'all_original_analysis_nodes_completed': len(result['nodes']),
          'decision_called': bool(result['decision_node_outputs']), 'actual_environment_submission_count': len(environment.actions),
          'proposal_has_no_observation': True, 'graph_termination': result['termination']}
path = Path('results/validation/agentprune_protocol_repair_20260924/replay_regression.json')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
