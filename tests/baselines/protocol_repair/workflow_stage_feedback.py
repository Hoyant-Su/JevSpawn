import argparse
import json
from pathlib import Path
from unittest.mock import Mock

from baselines.common.config import SharedConfig
from baselines.common.dyflow_v3 import load_core
from baselines.common.errors import InputLimitError, InvalidOutputError, TaskLimitError


def run(configuration):
    shared = SharedConfig.load(configuration['shared_config'])
    method = json.loads(Path(configuration['method']).read_text())
    settings = shared.method_settings(method['settings'])
    settings['workflow_stages'] = settings['max_turns']
    core = load_core(settings, Mock())
    designer = Mock()
    designer.generate.return_value = {'response': configuration['response']}
    workflow = core['WorkflowExecutor'](configuration['problem'], designer, Mock(), save_design_history=True)
    workflow._extract_json_from_string = Mock(side_effect=[
        InvalidOutputError(configuration['format_error']), configuration['stage']])
    workflow._execute_stage = Mock(return_value=(False, configuration['termination'], configuration['answer']))
    assert workflow.execute() == configuration['answer']
    assert configuration['format_error'] in designer.generate.call_args.kwargs['prompt']
    assert len(workflow.state.error_log) == configuration['expected_errors']
    assert designer.generate.call_count == configuration['expected_calls']
    assert workflow._execute_stage.call_count == configuration['expected_successful_stages']
    propagated = []
    for error_type in (TaskLimitError, InputLimitError, TimeoutError, RuntimeError):
        workflow = core['WorkflowExecutor'](configuration['problem'], Mock(), Mock(), save_design_history=True)
        failure = error_type(configuration['required_failure'])
        workflow._design_next_stage = Mock(side_effect=failure)
        try:
            workflow.execute()
        except error_type as error:
            assert error is failure
        else:
            raise AssertionError('A required failure was swallowed.')
        propagated.append(error_type.__name__)
    workflow = core['WorkflowExecutor'](configuration['problem'], Mock(), Mock(), save_design_history=True)
    workflow._design_next_stage = Mock(side_effect=InvalidOutputError(configuration['format_error']))
    assert workflow.execute() is None
    assert workflow._design_next_stage.call_count == settings['max_turns']
    result = {'status': 'passed', 'scope': 'Official workflow execution with mocked model transport; no task accuracy claim.',
              'format_error_visible_to_next_stage': True, 'no_substitute_stage_or_answer': True,
              'required_failures_propagate': propagated, 'stage_budget': settings['max_turns']}
    Path(configuration['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
