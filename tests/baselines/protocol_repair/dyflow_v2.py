import json
from pathlib import Path

from baselines.common import dyflow as original
from dyflow_candidate import load_core as initial_load_core


TEMPLATES = json.loads((Path(__file__).parent / 'template/prompts_v2.json').read_text())['dyflow_v2']


class BoundObservations:
    def __init__(self, environment):
        self.environment = environment

    def __getattr__(self, name):
        return getattr(self.environment, name)

    def execute(self, name, arguments):
        observation, done = self.environment.execute(name, arguments)
        if name != TEMPLATES['submission_tool']:
            observation = json.dumps({'tool': name, 'arguments': arguments,
                                      'observation': observation}, ensure_ascii=False)
        return observation, done


def load_core(settings, service_factory):
    core = initial_load_core(settings, service_factory)
    original_operator = core['InstructExecutorOperator']
    original_workflow = core['WorkflowExecutor']
    core['DESIGN_STAGE_PROMPT'] = core['DESIGN_STAGE_PROMPT'].replace(
        TEMPLATES['upstream_stage_limit'], TEMPLATES['stage_limit'].format(max_stages=settings['workflow_stages']))

    class CompleteContextOperator(original_operator):
        def _build_context_string(self, context):
            return super()._build_context_string({**context, 'original_problem': self.task_interface})

    class RuntimeWorkflow(original_workflow):
        def _design_next_stage(self):
            summary = self.state.get_state_summary_for_designer()
            prompt = core['DESIGN_STAGE_PROMPT'].format(
                problem_description=self.state.original_problem, state_summary=summary)
            prompt += TEMPLATES['stage_budget'].format(completed_stages=len(self.state.stages),
                                                       max_stages=settings['workflow_stages'])
            output = self.designer_llm.generate(prompt=prompt, temperature=settings['temperature'])['response']
            stage = self._extract_json_from_string(output)
            self.state.add_stage(stage['stage_description'], stage_id=stage['stage_id'])
            self.design_history.append({'input': prompt, 'output': output})
            return stage

    core['InstructExecutorOperator'] = CompleteContextOperator
    core['WorkflowExecutor'] = RuntimeWorkflow
    return core


original.load_core = load_core


def solve(task, environment, complete, settings, prompts):
    return original.solve(task, BoundObservations(environment), complete, settings, prompts)
