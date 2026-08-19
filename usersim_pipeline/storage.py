"""Persist complete experiment records as JSON."""

from pathlib import Path

from .schemas import ExperimentRun


def save_run(run: ExperimentRun, directory: str | Path = "runs") -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{run.run_id}.json"
    path.write_text(run.to_json(indent=2), encoding="utf-8")
    return path
