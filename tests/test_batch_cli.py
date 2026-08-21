import json

from usersim_pipeline.batch_cli import (
    BatchTask,
    _is_matching_completed_run,
    _manifest_matches,
    build_batch_plan,
    calls_per_run,
    questions_for_task,
    select_target_pairs,
)
from usersim_pipeline.questions import QUESTION_POOL
from usersim_pipeline.target_pairs import TargetPair


def pair(index):
    return TargetPair(
        pair_id=f"PAIR_{index:03d}",
        food_category_code=f"FOOD_{index}",
        food_category=f"Food {index}",
        waste_reason_code=f"REASON_{index}",
        waste_reason=f"Reason {index}",
    )


def test_batch_plan_has_expected_full_experiment_call_counts():
    plan = build_batch_plan(100, ["single", "multi"])

    assert calls_per_run("single") == 1
    assert calls_per_run("multi") == len(QUESTION_POOL) + 2 == 33
    assert plan["runs_by_mode"] == {"single": 100, "multi": 100}
    assert plan["api_calls_by_mode"] == {"single": 100, "multi": 3300}
    assert plan["total_runs"] == 200
    assert plan["total_api_calls"] == 3400


def test_resume_manifest_is_bound_to_exact_pair_selection():
    manifest = {
        "dataset_sha256": "abc",
        "model": "openai/gpt-5-nano",
        "modes": ["single"],
        "pair_ids": ["PAIR_001"],
        "selection": "random",
        "sample_seed": 42,
        "multi_question_order": None,
        "question_order_seed": None,
        "question_count": len(QUESTION_POOL),
    }

    assert _manifest_matches(
        manifest,
        dataset_sha256="abc",
        model_name="openai/gpt-5-nano",
        modes=["single"],
        pair_ids=["PAIR_001"],
        selection="random",
        sample_seed=42,
        question_order_seed=None,
    )
    assert not _manifest_matches(
        manifest,
        dataset_sha256="abc",
        model_name="openai/gpt-5-nano",
        modes=["single"],
        pair_ids=["PAIR_001", "PAIR_002"],
        selection="random",
        sample_seed=42,
        question_order_seed=None,
    )


def test_select_target_pairs_supports_first_and_reproducible_random_samples():
    pairs = [pair(index) for index in range(1, 11)]

    assert [item.pair_id for item in select_target_pairs(
        pairs, count=3, selection="first", seed=1
    )] == ["PAIR_001", "PAIR_002", "PAIR_003"]

    random_ids = [item.pair_id for item in select_target_pairs(
        pairs, count=3, selection="random", seed=42
    )]
    assert random_ids == [item.pair_id for item in select_target_pairs(
        pairs, count=3, selection="random", seed=42
    )]
    assert random_ids != ["PAIR_001", "PAIR_002", "PAIR_003"]


def test_multi_question_order_is_complete_pair_specific_and_reproducible():
    first_pair = pair(1)
    second_pair = pair(2)
    single_task = BatchTask(mode="single", pair=first_pair)
    first_multi_task = BatchTask(mode="multi", pair=first_pair)
    second_multi_task = BatchTask(mode="multi", pair=second_pair)
    canonical_ids = [question.id for question in QUESTION_POOL]

    assert [
        question.id
        for question in questions_for_task(single_task, question_order_seed=17)
    ] == canonical_ids

    first_order = [
        question.id
        for question in questions_for_task(first_multi_task, question_order_seed=17)
    ]
    repeated_order = [
        question.id
        for question in questions_for_task(first_multi_task, question_order_seed=17)
    ]
    second_order = [
        question.id
        for question in questions_for_task(second_multi_task, question_order_seed=17)
    ]

    assert first_order == repeated_order
    assert first_order != canonical_ids
    assert first_order != second_order
    assert set(first_order) == set(canonical_ids)
    assert len(first_order) == len(canonical_ids)


def test_completed_run_matching_accepts_compact_question_results(tmp_path):
    target = pair(1)
    task = BatchTask(mode="multi", pair=target)
    question_order_seed = 17
    question_ids = [
        question.id
        for question in questions_for_task(task, question_order_seed=question_order_seed)
    ]
    path = tmp_path / "PAIR_001.json"
    path.write_text(
        json.dumps(
            {
                "record_format": "cli_compact_v1",
                "mode": "multi",
                "target_pair_id": target.pair_id,
                "batch_id": "dataset_run_1",
                "food_category": target.food_category,
                "waste_reason": target.waste_reason,
                "model": "test/model",
                "question_order_strategy": "pair_seeded_shuffle",
                "question_order_seed": question_order_seed,
                "question_results": [
                    {"question_id": question_id} for question_id in question_ids
                ],
                "final_scenario": "Complete output.",
                "completed_at": "2026-08-21T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert _is_matching_completed_run(
        path,
        task=task,
        batch_id="dataset_run_1",
        model_name="test/model",
        question_order_seed=question_order_seed,
    )
