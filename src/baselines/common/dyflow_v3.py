import json
from pathlib import Path

from baselines.common import dyflow as original
from baselines.tool_agents.tools import ActionError


ORIGINAL_LOAD_CORE = original.load_core


class BoundObservations:
    def __init__(self, environment, submission_tool):
        self.environment = environment
        self.submission_tool = submission_tool

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def execute(self, name, arguments):
        observation, done = self.environment.execute(name, arguments)
        if name != self.submission_tool:
            observation = json.dumps({'tool': name, 'arguments': arguments,
                                      'observation': observation}, ensure_ascii=False)
        return observation, done


def load_core(settings, service_factory):
    templates = json.loads(Path(settings['protocol_templates']).read_text())
    core = ORIGINAL_LOAD_CORE(settings, service_factory)
    operator = core['InstructExecutorOperator']
    workflow = core['WorkflowExecutor']
    core['DESIGN_STAGE_PROMPT'] = core['DESIGN_STAGE_PROMPT'].replace(
        templates['upstream_stage_limit'], templates['stage_limit'].format(max_stages=settings['max_turns']))

    class ContextOperator(operator):
        def execute(self, state, params):
            self.task_interface = state.original_problem
            try:
                return super().execute(state, params)
            except ActionError:
                return 'error'

        def _build_context_string(self, context):
            return super()._build_context_string({**context, 'original_problem': self.task_interface})

        def _process_output(self, output, instruction):
            if instruction == templates['plan_operator']:
                before, separator, after = output.partition(templates['plan_prefix'])
                return {'content': after.strip() if separator else output}
            return super()._process_output(output, instruction)

    class RuntimeWorkflow(workflow):
        def _design_next_stage(self):
            summary = self.state.get_state_summary_for_designer()
            prompt = core['DESIGN_STAGE_PROMPT'].format(
                problem_description=self.state.original_problem, state_summary=summary)
            prompt += templates['execution_feedback'].format(errors=json.dumps(self.state.error_log, ensure_ascii=False))
            prompt += templates['stage_budget'].format(completed_stages=len(self.state.stages),
                                                        max_stages=settings['max_turns'])
            output = self.designer_llm.generate(prompt=prompt, temperature=settings['temperature'])['response']
            stage = self._extract_json_from_string(output)
            self.state.add_stage(stage['stage_description'], stage_id=stage['stage_id'])
            self.design_history.append({'input': prompt, 'output': output})
            return stage

    core['InstructExecutorOperator'] = ContextOperator
    core['WorkflowExecutor'] = RuntimeWorkflow
    return core


original.load_core = load_core


def solve(task, environment, complete, settings, prompts):
    templates = json.loads(Path(settings['protocol_templates']).read_text())
    settings = {**settings, 'workflow_stages': settings['max_turns']}
    return original.solve(task, BoundObservations(environment, templates['submission_tool']),
                          complete, settings, prompts)
