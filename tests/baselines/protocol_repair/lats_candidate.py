import json
from pathlib import Path

from baselines.common.lats import load_core as original_load_core, solve_with_core


TEMPLATES = json.loads((Path(__file__).parent / 'template/prompts.json').read_text())['lats']


class ChatTask:
    def __init__(self, task, prefix):
        self.task, self.prefix = task, prefix

    def __getattr__(self, name):
        return getattr(self.task, name)

    def cot_prompt_wrap(self, context, continuation, reflections):
        return self.task.cot_prompt_wrap(context, continuation, reflections) + TEMPLATES['chat_response'].format(prefix=self.prefix)


def load_core(source, gpt, environment):
    core, task = original_load_core(source, gpt, environment)
    original_samples = core['get_samples']

    def get_samples(task, context, prefix, count, prompt_sample, stop):
        return original_samples(ChatTask(task, prefix), context, '', count, prompt_sample, stop)

    core['get_samples'] = get_samples
    return core, task


def solve(task, environment, complete, settings, prompts):
    prompts = {**prompts, 'value_instruction': prompts['value_instruction'] + TEMPLATES['value_protocol']}
    return solve_with_core(task, environment, complete, settings, prompts, load_core)
