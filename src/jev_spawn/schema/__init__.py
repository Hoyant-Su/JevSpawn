from jev_spawn.infra.configuration import RESOURCES
from jev_spawn.infra.prompts import load_prompt


CONTROLLER = load_prompt(RESOURCES["controller"])


def controller_prefix(context, history):
    return CONTROLLER['prefix_template'].format(context=context, history=history)


def controller_prompts(states, question, options, labels, output_instruction, contexts=None, histories=None):
    contexts = [''] * len(states) if contexts is None else contexts
    histories = [''] * len(states) if histories is None else histories
    assert len(contexts) == len(states)
    menu = "\n".join(
        CONTROLLER["option_template"].format(label=label, **option)
        for label, option in zip(labels, options)
    )
    return [
        CONTROLLER["user_template"].format(
            prefix=controller_prefix(context, history), context=context, state=state, question=question, menu=menu,
            output_instruction=output_instruction
        )
        for context, state, history in zip(contexts, states, histories, strict=True)
    ]
