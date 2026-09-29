from datetime import datetime
from pathlib import Path


def create_run(root: Path, task_id: str, mode: str, now: datetime) -> Path:
    runs = Path(root) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stem = f"{task_id}__{mode}__{now.strftime('%Y%m%d_%H%M%S')}"
    candidate = runs / stem
    suffix = 1
    while candidate.exists():
        candidate = runs / f"{stem}__v{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate
