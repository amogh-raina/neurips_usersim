"""Persist complete experiment records as JSON."""

from pathlib import Path

from .schemas import ExperimentRun


def save_run(
    run: ExperimentRun,
    directory: str | Path = "runs",
    *,
    filename: str | None = None,
) -> Path:
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{run.run_id}.json"
    if Path(filename).name != filename:
        raise ValueError("Run filename must not contain directory components.")
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    path = output_dir / filename
    path.write_text(run.to_json(indent=2), encoding="utf-8")
    return path
