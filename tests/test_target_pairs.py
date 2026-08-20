import csv

import pytest

from usersim_pipeline.target_pairs import load_target_pairs


def test_load_target_pairs_reads_frozen_dataset():
    pairs = load_target_pairs()

    assert len(pairs) == 100
    assert pairs[0].pair_id == "PAIR_001"
    assert pairs[-1].pair_id == "PAIR_100"
    assert all(pair.food_category and pair.waste_reason for pair in pairs)


def test_load_target_pairs_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "pairs.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "pair_id",
                "food_category_code",
                "food_category",
                "waste_reason_code",
                "waste_reason",
            ],
        )
        writer.writeheader()
        row = {
            "pair_id": "PAIR_001",
            "food_category_code": "FOOD",
            "food_category": "Food",
            "waste_reason_code": "REASON",
            "waste_reason": "Reason",
        }
        writer.writerow(row)
        writer.writerow(row)

    with pytest.raises(ValueError, match="duplicate pair_id"):
        load_target_pairs(path)
