import re


def policy_sample(prefix, response, grammar):
    step = re.fullmatch(grammar['continuation_prefix'], prefix).group('step')
    complete = re.match(grammar['complete_frame'].format(step=step), response)
    return response if complete is not None else prefix + response
