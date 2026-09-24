from methods.source_interfaces.evaluate import evaluate, write_evaluation
from methods.source_sets.analysis import hierarchy_statistics, source_ids


if __name__ == '__main__':
    args, _, report = evaluate(hierarchy_statistics, source_ids)
    write_evaluation(args, report)
