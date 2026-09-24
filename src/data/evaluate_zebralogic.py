import argparse
import json
import re
from pathlib import Path


def score_grid(solution, prediction):
    columns, rows = solution["header"], solution["rows"]
    assert columns[0] == "House"
    total = len(rows) * (len(columns) - 1)
    parsed = isinstance(prediction, dict) and isinstance(prediction.get("solution"), dict)
    if not parsed:
        return {"correct_cells": 0, "total_cells": total, "solved": False, "parsed": False}
    grid = prediction["solution"]
    correct = 0
    for i, row in enumerate(rows):
        house = f"House {i + 1}"
        for j, column in enumerate(columns[1:], start=1):
            if house not in grid or column not in grid[house]:
                continue
            value = grid[house][column]
            if value is None or value == "" or value == []:
                continue
            if isinstance(value, list):
                value = value[0]
            if not isinstance(value, str):
                raise ValueError(f"Unsupported predicted cell type: {type(value).__name__}")
            correct += row[j].lower().strip() == value.lower().strip()
    return {"correct_cells": correct, "total_cells": total,
            "solved": correct == total, "parsed": True}


def source_column_mapping(puzzle, solution):
    preamble = puzzle.split("## Clues:")[0]
    source_sets = [set(value.lower().strip() for value in re.findall(r"`([^`]+)`", line.split(":", 1)[1]))
                   for line in preamble.splitlines() if line.startswith(" - ")]
    gold_sets = [set(row[index].lower().strip() for row in solution["rows"])
                 for index in range(1, len(solution["header"]))]
    matches = [[index for index, values in enumerate(gold_sets) if values == source]
               for source in source_sets]
    assert len(source_sets) == len(gold_sets) and all(len(indices) == 1 for indices in matches)
    mapping = [indices[0] for indices in matches]
    assert len(set(mapping)) == len(mapping)
    return mapping


def score_rows(puzzle, solution, predictionrows):
    mapping = source_column_mapping(puzzle, solution)
    valid = (isinstance(predictionrows, list) and len(predictionrows) == len(solution["rows"])
             and all(isinstance(row, list) and len(row) == len(mapping)
                     and all(isinstance(value, str) for value in row) for row in predictionrows))
    if not valid:
        return {**score_grid(solution, None), "shape_valid": False}
    grid = {f"House {i + 1}": {solution["header"][mapping[j] + 1]: value
                              for j, value in enumerate(row)} for i, row in enumerate(predictionrows)}
    return {**score_grid(solution, {"solution": grid}), "shape_valid": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = [json.loads(line) for line in args.labels.read_text().splitlines()]
    outputs = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    predictions = {row["task_id"]: row["prediction"] for row in outputs}
    assert len(predictions) == len(outputs)
    assert set(predictions) == {row["task_id"] for row in labels}
    scores = [{"task_id": row["task_id"], "size": row["size"],
               **score_grid(row["solution"], predictions[row["task_id"]])} for row in labels]
    summary = {"tasks": len(scores), "solved": sum(row["solved"] for row in scores),
               "parsed": sum(row["parsed"] for row in scores),
               "puzzle_accuracy": sum(row["solved"] for row in scores) / len(scores),
               "cell_accuracy": sum(row["correct_cells"] for row in scores) / sum(row["total_cells"] for row in scores),
               "records": scores}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
