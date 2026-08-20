"""Resumable concurrent CLI for running the frozen target-pair dataset."""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Literal

from dotenv import load_dotenv

from .model_factory import create_chat_model, effective_reasoning_effort
from .questions import QUESTION_POOL
from .runners import run_multi_turn, run_single_turn
from .storage import save_run
from .target_pairs import DEFAULT_TARGET_PAIRS_PATH, TargetPair, load_target_pairs

BatchMode = Literal["single", "multi"]
PairSelection = Literal["first", "random"]


@dataclass(frozen=True)
class BatchTask:
    mode: BatchMode
    pair: TargetPair


@dataclass(frozen=True)
class TaskResult:
    mode: BatchMode
    pair_id: str
    status: Literal["completed", "failed"]
    output_path: str | None
    completed_calls: int
    scenario_word_count: int | None = None
    scenario_word_count_valid: bool | None = None
    error_type: str | None = None
    error_message: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_modes(mode: str) -> list[BatchMode]:
    if mode == "both":
        return ["single", "multi"]
    if mode == "single":
        return ["single"]
    return ["multi"]


def calls_per_run(mode: BatchMode) -> int:
    return 1 if mode == "single" else len(QUESTION_POOL) + 2


def build_batch_plan(pair_count: int, modes: list[BatchMode]) -> dict[str, Any]:
    runs_by_mode = {mode: pair_count for mode in modes}
    calls_by_mode = {
        mode: pair_count * calls_per_run(mode)
        for mode in modes
    }
    return {
        "pair_count": pair_count,
        "runs_by_mode": runs_by_mode,
        "total_runs": sum(runs_by_mode.values()),
        "calls_per_run": {mode: calls_per_run(mode) for mode in modes},
        "api_calls_by_mode": calls_by_mode,
        "total_api_calls": sum(calls_by_mode.values()),
    }


def select_target_pairs(
    pairs: list[TargetPair],
    *,
    count: int | None,
    selection: PairSelection,
    seed: int,
) -> list[TargetPair]:
    """Select all, the first N, or a reproducible random N target pairs."""
    selected_count = len(pairs) if count is None else count
    if selected_count < 1:
        raise ValueError("The number of pairs must be at least 1.")
    if selected_count > len(pairs):
        raise ValueError(
            f"Requested {selected_count} pairs, but the dataset contains only {len(pairs)}."
        )
    if selection == "first":
        return pairs[:selected_count]
    return random.Random(seed).sample(pairs, selected_count)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _run_path(batch_directory: Path, task: BatchTask) -> Path:
    return batch_directory / task.mode / f"{task.pair.pair_id}.json"


def _is_matching_completed_run(
    path: Path,
    *,
    task: BatchTask,
    batch_id: str,
    model_name: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        (
            value.get("mode") == task.mode,
            value.get("target_pair_id") == task.pair.pair_id,
            value.get("batch_id") == batch_id,
            value.get("food_category") == task.pair.food_category,
            value.get("waste_reason") == task.pair.waste_reason,
            value.get("model") == model_name,
            bool(value.get("final_scenario")),
            value.get("completed_at") is not None,
        )
    )


def _execute_task(
    task: BatchTask,
    *,
    batch_id: str,
    batch_directory: Path,
    model_name: str,
    request_timeout: float | None,
) -> TaskResult:
    completed_calls = 0

    def count_completed_call(*_: Any) -> None:
        nonlocal completed_calls
        completed_calls += 1

    try:
        session_id = f"usersim-{batch_id}-{task.mode}-{task.pair.pair_id}"
        model = create_chat_model(
            model_name,
            session_id=session_id,
            timeout_seconds=request_timeout,
        )
        runner = run_single_turn if task.mode == "single" else run_multi_turn
        run = runner(
            model,
            food_category=task.pair.food_category,
            waste_reason=task.pair.waste_reason,
            questions=QUESTION_POOL,
            model_name=model_name,
            reasoning_effort=effective_reasoning_effort(model),
            target_pair_id=task.pair.pair_id,
            batch_id=batch_id,
            on_turn=count_completed_call,
        )
        output_path = save_run(
            run,
            batch_directory / task.mode,
            filename=task.pair.pair_id,
        )
        return TaskResult(
            mode=task.mode,
            pair_id=task.pair.pair_id,
            status="completed",
            output_path=str(output_path),
            completed_calls=completed_calls,
            scenario_word_count=run.final_scenario_word_count,
            scenario_word_count_valid=run.final_scenario_word_count_valid,
        )
    except Exception as exc:
        return TaskResult(
            mode=task.mode,
            pair_id=task.pair.pair_id,
            status="failed",
            output_path=None,
            completed_calls=completed_calls,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen target-pair dataset through SINGLE, MULTI, or both pipelines."
    )
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="both")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_TARGET_PAIRS_PATH)
    parser.add_argument("--model", default=os.getenv("USERSIM_MODEL", "openai/gpt-5-nano"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Parent directory; results are stored in dataset_run_<N> beneath it.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument(
        "--num-pairs",
        "--limit",
        dest="num_pairs",
        type=int,
        help="Number of dataset rows to run; defaults to all rows.",
    )
    parser.add_argument(
        "--selection",
        choices=["first", "random"],
        default="first",
        help="Choose the first N rows or a reproducible random sample.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260819,
        help="Seed used with --selection random.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.num_pairs is not None and args.num_pairs < 1:
        parser.error("--num-pairs must be at least 1")
    if args.request_timeout is not None and args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than zero")


def _manifest_matches(
    manifest: dict[str, Any],
    *,
    dataset_sha256: str,
    model_name: str,
    modes: list[BatchMode],
    pair_ids: list[str],
    selection: PairSelection,
    sample_seed: int | None,
) -> bool:
    return all(
        (
            manifest.get("dataset_sha256") == dataset_sha256,
            manifest.get("model") == model_name,
            manifest.get("modes") == modes,
            manifest.get("pair_ids") == pair_ids,
            manifest.get("selection") == selection,
            manifest.get("sample_seed") == sample_seed,
            manifest.get("question_count") == len(QUESTION_POOL),
        )
    )


def main() -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    dataset_path = args.dataset.resolve()
    all_pairs = load_target_pairs(dataset_path)
    try:
        pairs = select_target_pairs(
            all_pairs,
            count=args.num_pairs,
            selection=args.selection,
            seed=args.sample_seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    modes = _selected_modes(args.mode)
    plan = build_batch_plan(len(pairs), modes)
    batch_id = f"dataset_run_{len(pairs)}"
    batch_directory = args.output_dir / batch_id
    manifest_path = batch_directory / "manifest.json"
    dataset_sha256 = _dataset_sha256(dataset_path)

    summary = {
        "batch_id": batch_id,
        "output_directory": str(batch_directory),
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "model": args.model,
        "modes": modes,
        "pair_ids": [pair.pair_id for pair in pairs],
        "selection": args.selection,
        "sample_seed": args.sample_seed if args.selection == "random" else None,
        "concurrency": args.concurrency,
        "request_timeout_seconds": args.request_timeout
        if args.request_timeout is not None
        else float(os.getenv("USERSIM_REQUEST_TIMEOUT", "120")),
        **plan,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    if not os.getenv("OPENROUTER_API_KEY"):
        parser.error(
            "OPENROUTER_API_KEY is not set. Add it to .env or the shell environment."
        )

    if batch_directory.exists() and not args.resume:
        parser.error(
            f"Run directory already exists: {batch_directory}. Use --resume to continue the same "
            "selection, or choose a different --output-dir for a separate run."
        )

    if args.resume:
        if not manifest_path.is_file():
            parser.error(f"Cannot resume without manifest: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not _manifest_matches(
            existing_manifest,
            dataset_sha256=dataset_sha256,
            model_name=args.model,
            modes=modes,
            pair_ids=summary["pair_ids"],
            selection=args.selection,
            sample_seed=summary["sample_seed"],
        ):
            parser.error("Resume settings do not match the existing batch manifest.")

    all_tasks = [BatchTask(mode=mode, pair=pair) for mode in modes for pair in pairs]
    completed_tasks = [
        task
        for task in all_tasks
        if _is_matching_completed_run(
            _run_path(batch_directory, task),
            task=task,
            batch_id=batch_id,
            model_name=args.model,
        )
    ]
    completed_keys = {(task.mode, task.pair.pair_id) for task in completed_tasks}
    pending_tasks = [
        task
        for task in all_tasks
        if (task.mode, task.pair.pair_id) not in completed_keys
    ]

    previous_manifest = existing_manifest if args.resume else {}
    manifest: dict[str, Any] = {
        **summary,
        "question_count": len(QUESTION_POOL),
        "created_at": previous_manifest.get("created_at", _utc_now()),
        "updated_at": _utc_now(),
        "status": "running" if pending_tasks else "completed",
        "completed_runs": [f"{task.mode}:{task.pair.pair_id}" for task in completed_tasks],
        "failed_runs": previous_manifest.get("failed_runs", {}),
        "completed_model_calls": previous_manifest.get("completed_model_calls", 0),
    }
    _atomic_write_json(manifest_path, manifest)

    if not pending_tasks:
        print(f"Batch {batch_id} is already complete: {batch_directory}")
        return

    print(json.dumps({**summary, "pending_runs": len(pending_tasks)}, indent=2), flush=True)
    failures = 0
    completed_model_calls = int(manifest["completed_model_calls"])
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        future_to_task: dict[Future[TaskResult], BatchTask] = {
            executor.submit(
                _execute_task,
                task,
                batch_id=batch_id,
                batch_directory=batch_directory,
                model_name=args.model,
                request_timeout=args.request_timeout,
            ): task
            for task in pending_tasks
        }
        for index, future in enumerate(as_completed(future_to_task), 1):
            result = future.result()
            completed_model_calls += result.completed_calls
            key = f"{result.mode}:{result.pair_id}"
            if result.status == "completed":
                manifest["completed_runs"].append(key)
                manifest["failed_runs"].pop(key, None)
                length_status = (
                    "within range" if result.scenario_word_count_valid else "outside range"
                )
                print(
                    f"[{index}/{len(pending_tasks)}] {key} completed; "
                    f"{result.scenario_word_count} words ({length_status})",
                    flush=True,
                )
            else:
                failures += 1
                manifest["failed_runs"][key] = asdict(result)
                print(
                    f"[{index}/{len(pending_tasks)}] {key} failed: "
                    f"{result.error_type}: {result.error_message}",
                    flush=True,
                )
            manifest["completed_model_calls"] = completed_model_calls
            manifest["updated_at"] = _utc_now()
            manifest["status"] = "running"
            _atomic_write_json(manifest_path, manifest)

    manifest["updated_at"] = _utc_now()
    manifest["status"] = "completed" if failures == 0 else "completed_with_failures"
    _atomic_write_json(manifest_path, manifest)
    print(f"Batch output: {batch_directory}", flush=True)
    if failures:
        print(
            f"{failures} run(s) failed. Re-run the same command with --resume.",
            flush=True,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
