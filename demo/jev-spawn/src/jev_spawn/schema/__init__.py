import json
from importlib.resources import files


SYSTEM = json.loads(files(__package__).joinpath("roles.json").read_text())
DECISIONS = json.loads(files(__package__).joinpath("decisions.json").read_text())
CONTROLLER = json.loads(files(__package__).joinpath("controller.json").read_text())
BENCHMARK = json.loads(files(__package__).joinpath("benchmark.json").read_text())


def controller_prompts(states, question, options, labels, output_instruction):
    menu = "\n".join(
        CONTROLLER["option_template"].format(label=label, **option)
        for label, option in zip(labels, options)
    )
    return [
        CONTROLLER["user_template"].format(
            state=state, question=question, menu=menu, output_instruction=output_instruction
        )
        for state in states
    ]

SPECIALISTS = json.loads(files(__package__).joinpath("specialists.json").read_text())


def joint_controller_prompts(states, fields, labels, output_instruction=BENCHMARK["joint_output_instruction"]):
    descriptions = []
    for name, field in fields.items():
        menu = "\n".join(CONTROLLER["option_template"].format(label=label, **option)
                         for label, option in zip(labels, field["options"]))
        descriptions.append(BENCHMARK["joint_field_template"].format(
            name=name, question=field["question"], menu=menu))
    return [BENCHMARK["joint_user_template"].format(
        state=state, fields="\n\n".join(descriptions), output_instruction=output_instruction)
        for state in states]
