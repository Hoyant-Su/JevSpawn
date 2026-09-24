from jev_spawn.schema import CONTROLLER
from jev_spawn.algo.structured import score_fields


def score_grouped(backend, groups, mode, *, prefix_cache=None, prefix_lengths=None,
                  base_prefix_cache=None, base_prefix_length=None, admitted_prompts=None):
    assert groups and all(groups)
    assert len(groups) <= backend.config["batch_size"]
    assert all(len({node["id"] for node in group}) == len(group) for group in groups)
    assert "{id}" not in CONTROLLER["option_template"], "Positional option IDs require the declared label-description renderer."
    schemas = [{f"field{index}": {"question": node["question"],
                                "options": [{**option, "id": f"option{slot}"}
                                            for slot, option in enumerate(node["options"])]}
                for index, node in enumerate(group)} for group in groups]
    result = score_fields(backend, [group[0]["state"] for group in groups], schemas, mode,
                          field_states=[[node["state"] for node in group] for group in groups],
                          prefix_cache=prefix_cache, prefix_lengths_override=prefix_lengths,
                          base_prefix_cache=base_prefix_cache, base_prefix_length=base_prefix_length,
                          admitted_prompts=admitted_prompts)
    answers = []
    for row, group in enumerate(groups):
        output = []
        for index, node in enumerate(group):
            field = result["fields"][f"field{index}"]
            choice = field["option_ids"].index(field["choices"][row])
            output.append({"id": node["id"], "choice": node["options"][choice]["id"],
                           "probabilities": field["probabilities"][row],
                           "option_logits": field["option_logits"][row],
                           "option_ids": [option["id"] for option in node["options"]],
                           "input_tokens": field["input_tokens"][row]})
        answers.append(output)
    count = sum(map(len, groups))
    result.update(groups=answers, root_batch_size=len(groups), group_sizes=list(map(len, groups)),
                  logical_field_count=count, branch_tile_size=backend.config["branch_batch_size"],
                  option_counts=[[len(node["options"]) for node in group] for group in groups],
                  peak_field_concurrency=min(count, backend.config["branch_batch_size"])
                  if mode in {"streamed", "tiled_shared", "tiled_independent"} else count)
    return result
