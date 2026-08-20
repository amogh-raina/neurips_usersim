from usersim_pipeline.batch_cli import (
    _manifest_matches,
    build_batch_plan,
    calls_per_run,
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
    )
    assert not _manifest_matches(
        manifest,
        dataset_sha256="abc",
        model_name="openai/gpt-5-nano",
        modes=["single"],
        pair_ids=["PAIR_001", "PAIR_002"],
        selection="random",
        sample_seed=42,
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
