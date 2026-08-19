from copy import deepcopy

from usersim_pipeline.questions import ScenarioQuestion
from usersim_pipeline.runners import run_multi_turn, run_single_turn
from usersim_pipeline.text_metrics import (
    MAX_SCENARIO_WORDS,
    MIN_SCENARIO_WORDS,
    count_words,
    scenario_word_count_is_valid,
)
from langchain.messages import AIMessage


class StructuredInvoker:
    def __init__(self, parent, schema):
        self.parent = parent
        self.schema = schema

    def invoke(self, messages):
        self.parent.calls.append(deepcopy(messages))
        title = self.schema["title"]
        call_number = len(self.parent.calls)
        if title == "SingleTurnScenario":
            question_ids = self.schema["properties"]["question_answers"]["required"]
            parsed = {
                "focal_food": "carrots",
                "starting_point": "A household planned to use carrots.",
                "question_answers": {
                    question_id: f"answer-{question_id}"
                    for question_id in question_ids
                },
                "final_scenario": "single final scenario",
            }
        elif title == "InitialScenarioState":
            parsed = {"focal_food": "bread", "starting_point": "response-1"}
        elif title == "HouseholdQuestionAnswer":
            parsed = {"answer": f"response-{call_number}"}
        else:
            parsed = {"final_scenario": f"response-{call_number}"}
        raw = AIMessage(
            content=str(parsed),
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )
        return {"raw": raw, "parsed": parsed, "parsing_error": None}


class RecordingModel:
    def __init__(self):
        self.calls = []
        self.schemas = []

    def with_structured_output(self, schema, **kwargs):
        self.schemas.append((schema, kwargs))
        return StructuredInvoker(self, schema)


def question(question_id, text, aspect="Test aspect"):
    return ScenarioQuestion(
        id=question_id,
        aspect_id=1,
        aspect=aspect,
        text=text,
    )


def test_word_count_is_deterministic_and_treats_compounds_as_one_word():
    text = "A household's well-planned food-waste routine changed overnight."

    assert count_words(text) == 7
    assert count_words(text) == count_words(text)
    assert scenario_word_count_is_valid(MIN_SCENARIO_WORDS)
    assert scenario_word_count_is_valid(MAX_SCENARIO_WORDS)
    assert not scenario_word_count_is_valid(MIN_SCENARIO_WORDS - 1)
    assert not scenario_word_count_is_valid(MAX_SCENARIO_WORDS + 1)


def test_single_turn_makes_one_structured_call_with_all_inputs():
    model = RecordingModel()
    run = run_single_turn(
        model,
        food_category="vegetables",
        waste_reason="changed plans",
        questions=[question("Q1", "Question one?"), question("Q2", "Question two?")],
        model_name="test/model",
        reasoning_effort=None,
    )

    assert len(model.calls) == 1
    prompt = model.calls[0][-1].content
    assert "vegetables" in prompt
    assert "changed plans" in prompt
    assert "Q1. Question one?" in prompt
    assert "Q2. Question two?" in prompt
    assert "between 200 and\n350 words, inclusive" in prompt
    assert "200-350 word scenario" in model.schemas[0][0]["properties"]["final_scenario"]["description"]
    assert model.schemas[0][1]["method"] == "json_schema"
    assert model.schemas[0][1]["strict"] is True
    answers_schema = model.schemas[0][0]["properties"]["question_answers"]
    assert answers_schema["type"] == "object"
    assert answers_schema["required"] == ["Q1", "Q2"]
    assert answers_schema["additionalProperties"] is False
    assert run.final_scenario == "single final scenario"
    assert run.final_scenario_word_count == 3
    assert run.final_scenario_word_count_valid is False
    assert run.turns[0].scenario_word_count == 3
    assert run.turns[0].scenario_word_count_valid is False
    assert len(run.turns[0].structured_response["question_answers"]) == 2
    assert run.question_results[0].question_id == "Q1"
    assert run.question_results[0].aspect == "Test aspect"
    assert run.question_results[0].answer == "answer-Q1"


def test_multi_turn_replays_history_and_keeps_aspect_metadata_out_of_prompt():
    model = RecordingModel()
    run = run_multi_turn(
        model,
        food_category="bread",
        waste_reason="forgotten",
        questions=[
            question("Q1", "Question one?", "Aspect one"),
            question("Q2", "Question two?", "Aspect two"),
        ],
        model_name="test/model",
        reasoning_effort="low",
    )

    assert len(model.calls) == 4
    assert len(model.calls[0]) == 2
    assert len(model.calls[1]) == 3
    assert len(model.calls[2]) == 5
    assert len(model.calls[3]) == 7

    initial_history = [message.content for message in model.calls[0]]
    first_question_history = [message.content for message in model.calls[1]]
    assert "forgotten" in initial_history[-1]
    assert "forgotten" not in first_question_history
    assert "Do not yet describe" not in first_question_history
    assert "Focal food: bread" in first_question_history[1]
    assert "Starting point: response-1" in first_question_history[1]
    assert "scenario" not in first_question_history[0].lower()

    final_history = [message.content for message in model.calls[-1]]
    assert "Question one?" in final_history
    assert "response-2" in final_history
    assert "Question two?" in final_history
    assert "response-3" in final_history
    assert "Aspect one" not in final_history
    assert "Aspect two" not in final_history
    assert "producing a final household food-waste scenario" in final_history[0]
    assert "between 200 and 350\nwords, inclusive" in final_history[0]
    assert "answering a sequence of questions" not in final_history[0]
    assert "200-350 word scenario" in model.schemas[-1][0]["properties"]["final_scenario"]["description"]
    assert run.turns[1].question_id == "Q1"
    assert run.turns[1].aspect == "Aspect one"
    assert run.turns[1].system_prompt == run.system_prompts["question"]
    assert run.turns[-1].system_prompt == run.system_prompts["final"]
    assert run.question_context.startswith("The initialization phase established")
    assert run.reasoning_effort == "low"
    assert run.question_results[1].aspect == "Aspect two"
    assert run.question_results[1].answer == "response-3"
    assert run.final_scenario == "response-4"
    assert run.final_scenario_word_count == 1
    assert run.final_scenario_word_count_valid is False
    assert run.turns[-1].scenario_word_count == 1
    assert run.turns[-1].scenario_word_count_valid is False
    assert run.turns[1].scenario_word_count is None
    assert run.turns[1].scenario_word_count_valid is None
