"""Load and validate frozen food-category/waste-reason target pairs."""

import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_TARGET_PAIRS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "finnish_household_food_waste_target_pairs.csv"
)

REQUIRED_COLUMNS = {
    "pair_id",
    "food_category_code",
    "food_category",
    "waste_reason_code",
    "waste_reason",
}


@dataclass(frozen=True)
class TargetPair:
    pair_id: str
    food_category_code: str
    food_category: str
    waste_reason_code: str
    waste_reason: str


def load_target_pairs(path: str | Path = DEFAULT_TARGET_PAIRS_PATH) -> list[TargetPair]:
    dataset_path = Path(path)
    with dataset_path.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(f"Target-pair dataset is missing columns: {missing_columns}")
        rows = list(reader)

    pairs = [
        TargetPair(
            pair_id=row["pair_id"].strip(),
            food_category_code=row["food_category_code"].strip(),
            food_category=row["food_category"].strip(),
            waste_reason_code=row["waste_reason_code"].strip(),
            waste_reason=row["waste_reason"].strip(),
        )
        for row in rows
    ]
    if not pairs:
        raise ValueError("Target-pair dataset is empty.")

    for pair in pairs:
        if not all(
            (
                pair.pair_id,
                pair.food_category_code,
                pair.food_category,
                pair.waste_reason_code,
                pair.waste_reason,
            )
        ):
            raise ValueError(f"Target-pair dataset contains an incomplete row: {pair}")

    pair_ids = [pair.pair_id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Target-pair dataset contains duplicate pair_id values.")
    return pairs
