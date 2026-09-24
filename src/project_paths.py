from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def log_path(output, filename):
    path = ROOT / 'logs' / Path(output).resolve().relative_to(ROOT) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
