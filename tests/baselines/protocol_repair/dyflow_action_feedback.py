import argparse
import json
from pathlib import Path
from unittest.mock import Mock

from baselines.common.config import SharedConfig
from baselines.common.dyflow_v3 import load_core


def run(configuration):
    settings = json.loads(Path(configuration['method']).read_text())['settings']
    shared = SharedConfig.load(configuration['shared_config'])
    settings.update(workflow_stages=shared.runtime.max_turns, max_turns=shared.runtime.max_turns,
                    temperature=shared.generation.temperature)
    core = load_core(settings, Mock())
    state = core['State'](configuration['problem'])
    client = Mock()
    operator = core['InstructExecutorOperator'](
        configuration['operator_id'], configuration['operator_description'], client)
    cases = []
    for instruction in configuration['instructions']:
        params = {**configuration['parameters'], 'instruction_type': instruction}
        signal = operator.execute(state, params)
        assert signal == 'error'
        assert configuration['missing_path'] in state.error_log[-1]['message']
        cases.append({'instruction': instruction, 'signal': signal, 'error': state.error_log[-1]})
    client.generate.assert_not_called()
    assert state.actions == {} and state.final_answer is None
    designer = Mock()
    designer.generate.return_value = {'response': json.dumps(configuration['stage'])}
    workflow = core['WorkflowExecutor'](state.original_problem, designer, client, save_design_history=True)
    workflow.state = state
    assert workflow._design_next_stage() == configuration['stage']
    prompt = designer.generate.call_args.kwargs['prompt']
    assert all(error['message'] in prompt for error in state.error_log)
    client.generate.side_effect = RuntimeError(configuration['infra_failure'])
    try:
        operator.execute(state, configuration['valid_parameters'])
    except RuntimeError as error:
        assert str(error) == configuration['infra_failure']
    else:
        raise AssertionError('Infrastructure error was swallowed')
    result = {'scope': 'Upstream operator/workflow with simulated LLM transport, CPU only',
              'status': 'passed', 'recoverable_cases': cases,
              'error_visible_to_next_designer_request': True,
              'missing_result_substituted': False, 'infrastructure_error_propagates': True}
    Path(configuration['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
