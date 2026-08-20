"""Local records and provider-facing JSON Schemas."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from .questions import ScenarioQuestion
from .text_metrics import MAX_SCENARIO_WORDS, MIN_SCENARIO_WORDS


@dataclass
class TurnRecord:
    turn_index: int
    kind: Literal["single", "initial", "question", "final"]
    prompt: str
    system_prompt: str
    response: str
    structured_response: dict[str, Any]
    scenario_word_count: int | None = None
    scenario_word_count_valid: bool | None = None
    question_id: str | None = None
    aspect: str | None = None
    usage_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionResult:
    question_id: str
    aspect_id: int
    aspect: str
    question: str
    answer: str


@dataclass
class ExperimentRun:
    mode: Literal["single", "multi"]
    food_category: str
    waste_reason: str
    model: str
    reasoning_effort: str | None
    questions: list[ScenarioQuestion]
    system_prompts: dict[str, str]
    target_pair_id: str | None = None
    batch_id: str | None = None
    question_context: str = ""
    run_id: str = field(default_factory=lambda: str(uuid4()))
    turns: list[TurnRecord] = field(default_factory=list)
    question_results: list[QuestionResult] = field(default_factory=list)
    final_scenario: str = ""
    final_scenario_word_count: int = 0
    final_scenario_word_count_valid: bool = False
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["started_at"] = self.started_at.isoformat()
        value["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return value

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _object_schema(title: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "title": title,
        "description": description,
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


INITIAL_OUTPUT_SCHEMA = _object_schema(
    "InitialScenarioState",
    "The concrete starting state for a progressively developed food-waste situation.",
    {
        "focal_food": {
            "type": "string",
            "description": "The concrete focal food chosen within the requested category.",
        },
        "starting_point": {
            "type": "string",
            "description": "A concise 1-2 sentence starting situation.",
        },
    },
    ["focal_food", "starting_point"],
)

QUESTION_OUTPUT_SCHEMA = _object_schema(
    "HouseholdQuestionAnswer",
    "A concise answer to the current household food-waste question.",
    {
        "answer": {
            "type": "string",
            "description": "The concise answer, using the starting information and previous answers.",
        }
    },
    ["answer"],
)

FINAL_OUTPUT_SCHEMA = _object_schema(
    "FinalScenario",
    "The final synthesized household food-waste scenario.",
    {
        "final_scenario": {
            "type": "string",
            "description": (
                f"One coherent {MIN_SCENARIO_WORDS}-{MAX_SCENARIO_WORDS} word scenario using only "
                "established information."
            ),
        }
    },
    ["final_scenario"],
)


def single_output_schema(questions: list[ScenarioQuestion]) -> dict:
    question_ids = [question.id for question in questions]
    answer_properties = {
        question.id: {
            "type": "string",
            "description": "A concise answer consistent with the established situation.",
        }
        for question in questions
    }
    return _object_schema(
        "SingleTurnScenario",
        "A complete single-turn scenario construction with answers and final synthesis.",
        {
            "focal_food": {
                "type": "string",
                "description": "The concrete focal food chosen within the requested category.",
            },
            "starting_point": {
                "type": "string",
                "description": "The concise starting situation.",
            },
            "question_answers": {
                "type": "object",
                "description": "Exactly one concise answer keyed by every supplied question ID.",
                "properties": answer_properties,
                "required": question_ids,
                "additionalProperties": False,
            },
            "final_scenario": {
                "type": "string",
                "description": (
                    f"One coherent {MIN_SCENARIO_WORDS}-{MAX_SCENARIO_WORDS} word scenario using "
                    "only the established answers."
                ),
            },
        },
        ["focal_food", "starting_point", "question_answers", "final_scenario"],
    )
