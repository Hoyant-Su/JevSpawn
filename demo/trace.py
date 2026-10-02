ROUND_KEYS = (
    "turn", "remaining_control_rounds", "selection_only", "active_frontier",
    "ranked_compared", "selected", "selected_operation", "pruned_frontier",
    "retained_frontier", "parents",
)
DECISION_KEYS = ("id", "choice", "probabilities", "option_ids", "ranked_option_ids")


def fields(value, keys):
    return {key: value[key] for key in keys if key in value}


def compact_round(turn):
    result = fields(turn, ROUND_KEYS)
    for name in ("frontier_decision", "operation_decision"):
        result[name] = fields(turn[name], DECISION_KEYS)
    if "revision" in turn:
        result["revision"] = fields(
            turn["revision"], ("named_bindings", "declaration", "observation")
        )
    if "children" in turn:
        result["children"] = {
            node: [fields(step, (
                "actions", "observations", "parent_id", "selected_values",
                "block_conditional_log_probability",
            )) for step in steps]
            for node, steps in turn["children"].items()
        }
    if "parent_computations" in turn:
        result["parent_computations"] = {}
        for parent, computations in turn["parent_computations"].items():
            result["parent_computations"][parent] = []
            for computation in (item for item in computations if "decision" in item):
                decision = fields(computation["decision"], DECISION_KEYS)
                decision["options"] = [fields(option, ("id", "values"))
                                       for option in computation["decision"]["options"]]
                result["parent_computations"][parent].append({
                    "decision": decision,
                    "based_on_feedback": computation["based_on_feedback"],
                })
    if "submission" in turn:
        result["submission"] = fields(turn["submission"], (
            "branch_id", "history_event_ids", "mode", "answer",
            "resolved_arguments", "feedback",
        ))
    return result
