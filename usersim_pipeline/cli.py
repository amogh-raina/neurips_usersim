"""Command-line entry point for reproducible local runs."""

import argparse
import os

from dotenv import load_dotenv

from .model_factory import create_chat_model, effective_reasoning_effort
from .questions import QUESTION_POOL
from .runners import run_multi_turn, run_single_turn
from .storage import save_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a food-waste scenario experiment.")
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--food-category", required=True)
    parser.add_argument("--waste-reason", required=True)
    parser.add_argument("--model", default=os.getenv("USERSIM_MODEL", "openai/gpt-4.1-mini"))
    parser.add_argument("--output-dir", default="runs")
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    model = create_chat_model(args.model)
    runner = run_single_turn if args.mode == "single" else run_multi_turn
    run = runner(
        model,
        food_category=args.food_category,
        waste_reason=args.waste_reason,
        questions=QUESTION_POOL,
        model_name=args.model,
        reasoning_effort=effective_reasoning_effort(model),
        on_turn=lambda turn, _: print(f"[{turn.kind} {turn.turn_index}] {turn.response}\n"),
    )
    path = save_run(run, args.output_dir, compact=True)
    print(f"Saved run to {path}")


if __name__ == "__main__":
    main()
