from baselines.common.foldagent import original, solve_with_tokenizer


def solve(task, environment, complete, settings, prompts):
    tokenizer = original.TokenizerInterface(complete.func.__self__.host_tokenizer)
    return solve_with_tokenizer(task, environment, complete, settings, prompts, tokenizer)
