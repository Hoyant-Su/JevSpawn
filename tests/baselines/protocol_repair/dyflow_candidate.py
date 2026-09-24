import json
from pathlib import Path

from baselines.common import dyflow as original


TEMPLATES = json.loads((Path(__file__).parent / 'template/prompts.json').read_text())['dyflow']
ORIGINAL_LOAD_CORE = original.load_core


def load_core(settings, service_factory):
    core = ORIGINAL_LOAD_CORE(settings, service_factory)
    upstream = core['InstructExecutorOperator']

    class ContextOperator(upstream):
        def execute(self, state, params):
            self.task_interface = state.original_problem
            self.instruction = params['instruction_type']
            return super().execute(state, params)

        def _build_context_string(self, context):
            if self.instruction in TEMPLATES['interface_operators']:
                context = {**context, 'original_problem': self.task_interface}
            return super()._build_context_string(context)

        def _process_output(self, output, instruction):
            if instruction == TEMPLATES['plan_operator']:
                before, separator, after = output.partition(TEMPLATES['plan_prefix'])
                return {'content': after.strip() if separator else output}
            return super()._process_output(output, instruction)

    core['InstructExecutorOperator'] = ContextOperator
    return core


original.load_core = load_core
solve = original.solve
