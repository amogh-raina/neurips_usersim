import csv
from collections import Counter
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "finnish_household_food_waste_target_pairs.csv"
)

FOODS = {
    "VEGETABLES_POTATOES": ("Vegetables and potatoes", 19),
    "HOME_COOKED_FOOD": ("Home-cooked food", 18),
    "MILK_PRODUCTS": ("Milk products", 17),
    "FRUITS_BERRIES": ("Fruits and berries", 13),
    "BAKERY_GRAINS": ("Bakery and grain products", 13),
    "MEAT_FISH_EGGS": ("Meat, fish and eggs", 7),
    "CONVENIENCE_TAKEAWAY": ("Convenience and takeaway food", 6),
    "RICE_PASTA": ("Rice and pasta", 4),
    "OTHER_FOOD": ("Other food", 3),
}

REASONS = {
    "SPOILED_MOULDY": ("Spoiled or mouldy", 28),
    "DATE_EXPIRED": ("Best-before or use-by date expired", 19),
    "PLATE_LEFTOVERS": ("Plate leftovers", 14),
    "OVER_PREPARED": ("Over-prepared", 13),
    "NO_LONGER_WANTED": ("Food no longer wanted", 10),
    "SUSPECTED_PAST_BEST": ("Suspected to be past its best", 9),
    "OTHER_REASON": ("Other reason", 7),
}

EXPECTED_COLUMNS = [
    "pair_id",
    "food_category_code",
    "food_category",
    "waste_reason_code",
    "waste_reason",
    "food_is_residual",
    "reason_is_residual",
    "pairing_method",
    "joint_distribution_status",
    "random_seed",
]


def test_target_pair_dataset_is_complete_and_preserves_marginals():
    with DATASET_PATH.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        rows = list(reader)

    assert reader.fieldnames == EXPECTED_COLUMNS
    assert len(rows) == 100
    assert [row["pair_id"] for row in rows] == [
        f"PAIR_{index:03d}" for index in range(1, 101)
    ]

    assert Counter(row["food_category_code"] for row in rows) == Counter(
        {code: required_count for code, (_, required_count) in FOODS.items()}
    )
    assert Counter(row["waste_reason_code"] for row in rows) == Counter(
        {code: required_count for code, (_, required_count) in REASONS.items()}
    )

    for row in rows:
        assert row["food_category"] == FOODS[row["food_category_code"]][0]
        assert row["waste_reason"] == REASONS[row["waste_reason_code"]][0]
        assert (row["food_is_residual"] == "true") == (
            row["food_category_code"] == "OTHER_FOOD"
        )
        assert (row["reason_is_residual"] == "true") == (
            row["waste_reason_code"] == "OTHER_REASON"
        )
        assert row["pairing_method"] == "plausibility_constrained_pairing"
        assert (
            row["joint_distribution_status"]
            == "synthetic_not_empirically_observed"
        )
        assert row["random_seed"] == "20260819"
