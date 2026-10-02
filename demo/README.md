# JevSpawn demo

Explore the eight examples in the paper's Case Studies, one turn at a time. The page shows action declarations, spawned actions and their probabilities, environment observations, branch selection, revisions, and the submitted answer.

## Watch the recorded examples

From the repository root, run

```bash
bash demo/run.sh
```

Open **http://127.0.0.1:8765**. Python 3 is the only dependency for this mode. Select a task and press **Play trace**. Click a branch to inspect its action and returned observation, or expand a round to inspect the full record.

[Watch the browser recording](media/jevspawn.mp4)

Recorded mode displays complete saved inference traces at 0.9 seconds per round. Playback makes no model calls and does not represent inference latency. The final answers were re-executed with the original environment implementations. All 372 recorded branch actions were also re-executed and their observations matched.

| Task | Fixed case | Rounds | Executed branch actions | Recorded score |
| :--- | :--- | ---: | ---: | ---: |
| PPNL | ICL_test_set_moreobsts_updated/164 | 8 | 28 | 1 |
| Maze | lmrlgym_maze_10 | 11 | 40 | 1 |
| Grid | llfbench_gridworld_seed104 | 7 | 24 | 1 |
| LightsOut | textarena_lightsout_seed124 | 4 | 12 | 1 |
| RushHour | textarena_rushhour_seed138 | 14 | 52 | 1 |
| Sokoban | textarena_sokoban_seed88 | 7 | 24 | 1 |
| 2048 | korgym_2048_seed141 | 37 | 144 | 1076 |
| Nullify | korgym_nullify_seed129 | 13 | 48 | 1 |

2048 is a score-based task. Its result is shown as accumulated merge score. The final submission round can follow the configured exploration rounds.

## Run fresh GPU inference

Use four H100 80 GB GPUs and the inference dependencies in the repository's `pyproject.toml`. Set `model.path` in `configs/shared.yaml` to the Qwen3.8-27B checkpoint. The remaining shared and method settings are reused without changes.

Install the environment dependencies using the same Python environment as the inference runtime, then download the pinned upstream environment sources

```bash
python -m pip install -r demo/environments/requirements.txt
python -m demo.environments.setup
bash demo/run.sh --live
```

To select a Python executable explicitly

```bash
JEV_PYTHON=/path/to/environment/bin/python bash demo/run.sh --live
```

Choose **Live inference** in the page and press **Run inference**. Each run loads the model, executes the selected task through the public `jev_spawn.api.run` entry point, and streams newly completed rounds. Model loading precedes the first round. Only one live run uses the GPUs at a time. **Stop** terminates that demo run. New results can differ from the recorded examples.

Fresh inference receives the original task context. Saved answers and trajectories are used only in recorded mode and environment validation. Benchmark dispatch is confined to `environments/`; the JevSpawn algorithm receives no dataset identifier. Environment implementations and versions are pinned in `environments/sources.json` and `environments/requirements.txt`.

The eight-case live check on four H100 GPUs completed all tasks without interface errors. Six goal-based tasks scored 1, Nullify scored 0, and 2048 scored 328. Recorded outcomes above belong to the paper's selected trajectories, not this fresh run.

For a batch run of all eight fixed cases

```bash
python -m demo.launch --config demo/configs/demo.json --samples all --output demo/output/live
```

Use a new output directory for each batch run. Results, event streams, and worker logs stay under the ignored `demo/output/` directory. Server address and playback settings are defined in `demo/configs/demo.json`. Bind the demo to localhost and use port forwarding when running on a remote machine.

## Reproduce the checks and video

```bash
python -m demo.environments.validate --output demo/output/environment-check.json
python -m pip install playwright av
python -m playwright install chromium
bash demo/run.sh
```

With the server running, execute the following in another terminal

```bash
python -m demo.tools.record --config demo/configs/recording.json
```

The recording script captures the actual browser, checks all eight completed streams, and saves screenshots and the browser report under `demo/output/`. Video settings are explicit in `demo/configs/`. The recording is trace playback, clearly labeled throughout.

## Files

- `web/` — browser interface
- `server.py`, `launch.py`, `infer.py` — HTTP streaming and the public inference entry point
- `data/` — the eight complete paper cases
- `environments/` — official environment execution, evaluation, and pinned source acquisition
- `configs/` — demo and recording settings
- `trace.py` — compact turn events
- `tools/` — browser recording
- `media/` — demonstration video

The upstream sources retain their respective licenses. Sources are downloaded into the ignored `environments/external/` directory. TypeSafe Jev API access is not included.
