"""Matched SINGLE and MULTI pipelines with provider-enforced JSON outputs."""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from .prompts import (
    FINAL_SYSTEM_PROMPT,
    INITIAL_SYSTEM_PROMPT,
    MULTI_FINAL_PROMPT,
    QUESTION_SYSTEM_PROMPT,
    SINGLE_SYSTEM_PROMPT,
    multi_initial_prompt,
    multi_question_handoff,
    single_prompt,
)
from .questions import ScenarioQuestion
from .schemas import (
    FINAL_OUTPUT_SCHEMA,
    INITIAL_OUTPUT_SCHEMA,
    QUESTION_OUTPUT_SCHEMA,
    ExperimentRun,
    QuestionResult,
    TurnRecord,
    single_output_schema,
)
from .text_metrics import count_words, scenario_word_count_is_valid

TurnCallback = Callable[[TurnRecord, ExperimentRun], None]


def _usage(message: Any) -> dict[str, Any]:
    usage = getattr(message, "usage_metadata", None)
    return dict(usage) if usage else {}


def _normalize_questions(questions: Sequence[ScenarioQuestion | str]) -> list[ScenarioQuestion]:
    normalized: list[ScenarioQuestion] = []
    for index, question in enumerate(questions, 1):
        if isinstance(question, ScenarioQuestion):
            if question.text.strip():
                normalized.append(question)
        elif question.strip():
            normalized.append(
                ScenarioQuestion(
                    id=f"Q{index}",
                    aspect_id=0,
                    aspect="Custom",
                    text=question.strip(),
                )
            )
    return normalized


def _invoke_structured(model: Any, schema: dict, messages: list[Any]) -> tuple[Any, dict[str, Any]]:
    structured_model = model.with_structured_output(
        schema,
        method="json_schema",
        strict=True,
        include_raw=True,
    )
    result = structured_model.invoke(messages)
    parsing_error = result.get("parsing_error")
    if parsing_error:
        raise ValueError(f"The model returned an invalid structured response: {parsing_error}")
    parsed = result.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError("The model did not return the expected structured response.")
    return result.get("raw"), parsed


def _record(
    run: ExperimentRun,
    *,
    kind: str,
    prompt: str,
    system_prompt: str,
    response_text: str,
    structured_response: dict[str, Any],
    raw_response: Any,
    callback: TurnCallback | None,
    question: ScenarioQuestion | None = None,
) -> TurnRecord:
    scenario_word_count = count_words(response_text) if kind in {"single", "final"} else None
    turn = TurnRecord(
        turn_index=len(run.turns),
        kind=kind,
        prompt=prompt,
        system_prompt=system_prompt,
        response=response_text,
        structured_response=structured_response,
        scenario_word_count=scenario_word_count,
        scenario_word_count_valid=(
            scenario_word_count_is_valid(scenario_word_count)
            if scenario_word_count is not None
            else None
        ),
        question_id=question.id if question else None,
        aspect=question.aspect if question else None,
        usage_metadata=_usage(raw_response),
    )
    run.turns.append(turn)
    if callback:
        callback(turn, run)
    return turn


def _question_result(question: ScenarioQuestion, answer: str) -> QuestionResult:
    return QuestionResult(
        question_id=question.id,
        aspect_id=question.aspect_id,
        aspect=question.aspect,
        question=question.text,
        answer=answer,
    )


def _new_run(
    *,
    mode: str,
    food_category: str,
    waste_reason: str,
    questions: Sequence[ScenarioQuestion | str],
    model_name: str,
    reasoning_effort: str | None,
) -> ExperimentRun:
    normalized_questions = _normalize_questions(questions)
    if not food_category.strip() or not waste_reason.strip():
        raise ValueError("Food category and waste reason are required.")
    if not normalized_questions:
        raise ValueError("At least one scenario question is required.")
    return ExperimentRun(
        mode=mode,
        food_category=food_category.strip(),
        waste_reason=waste_reason.strip(),
        model=model_name,
        reasoning_effort=reasoning_effort,
        questions=normalized_questions,
        system_prompts=(
            {"single": SINGLE_SYSTEM_PROMPT}
            if mode == "single"
            else {
                "initial": INITIAL_SYSTEM_PROMPT,
                "question": QUESTION_SYSTEM_PROMPT,
                "final": FINAL_SYSTEM_PROMPT,
            }
        ),
    )


def run_single_turn(
    model: Any,
    *,
    food_category: str,
    waste_reason: str,
    questions: Sequence[ScenarioQuestion | str],
    model_name: str,
    reasoning_effort: str | None = None,
    on_turn: TurnCallback | None = None,
) -> ExperimentRun:
    """Make one call and require a structured starting point, answers, and scenario."""
    run = _new_run(
        mode="single",
        food_category=food_category,
        waste_reason=waste_reason,
        questions=questions,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
    prompt = single_prompt(run.food_category, run.waste_reason, run.questions)
    raw, parsed = _invoke_structured(
        model,
        single_output_schema(run.questions),
        [SystemMessage(SINGLE_SYSTEM_PROMPT), HumanMessage(prompt)],
    )
    returned_answers = parsed["question_answers"]
    if not isinstance(returned_answers, dict):
        raise ValueError("The single-turn question answers were not keyed by question ID.")
    answers_by_id = {str(question_id): str(answer) for question_id, answer in returned_answers.items()}
    expected_ids = {question.id for question in run.questions}
    if set(answers_by_id) != expected_ids:
        missing_ids = sorted(expected_ids - set(answers_by_id))
        unexpected_ids = sorted(set(answers_by_id) - expected_ids)
        raise ValueError(
            "The single-turn response did not contain exactly one answer per question ID. "
            f"Missing: {missing_ids or 'none'}; unexpected: {unexpected_ids or 'none'}."
        )
    run.question_results = [
        _question_result(question, answers_by_id[question.id])
        for question in run.questions
    ]
    final_scenario = str(parsed["final_scenario"])
    final_turn = _record(
        run,
        kind="single",
        prompt=prompt,
        system_prompt=SINGLE_SYSTEM_PROMPT,
        response_text=final_scenario,
        structured_response=parsed,
        raw_response=raw,
        callback=on_turn,
    )
    run.final_scenario = final_scenario
    run.final_scenario_word_count = final_turn.scenario_word_count or 0
    run.final_scenario_word_count_valid = bool(final_turn.scenario_word_count_valid)
    run.completed_at = datetime.now(timezone.utc)
    return run


def run_multi_turn(
    model: Any,
    *,
    food_category: str,
    waste_reason: str,
    questions: Sequence[ScenarioQuestion | str],
    model_name: str,
    reasoning_effort: str | None = None,
    on_turn: TurnCallback | None = None,
) -> ExperimentRun:
    """Invoke once per turn while replaying the complete prior Q/A history."""
    run = _new_run(
        mode="multi",
        food_category=food_category,
        waste_reason=waste_reason,
        questions=questions,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )
    initial_prompt = multi_initial_prompt(run.food_category, run.waste_reason)
    initial_messages: list[Any] = [
        SystemMessage(INITIAL_SYSTEM_PROMPT),
        HumanMessage(initial_prompt),
    ]

    raw, parsed = _invoke_structured(model, INITIAL_OUTPUT_SCHEMA, initial_messages)
    focal_food = str(parsed["focal_food"])
    starting_point = str(parsed["starting_point"])
    _record(
        run,
        kind="initial",
        prompt=initial_prompt,
        system_prompt=INITIAL_SYSTEM_PROMPT,
        response_text=starting_point,
        structured_response=parsed,
        raw_response=raw,
        callback=on_turn,
    )
    run.question_context = multi_question_handoff(focal_food, starting_point)
    question_history: list[Any] = [HumanMessage(run.question_context)]

    for question in run.questions:
        question_history.append(HumanMessage(question.text))
        messages = [SystemMessage(QUESTION_SYSTEM_PROMPT), *question_history]
        raw, parsed = _invoke_structured(model, QUESTION_OUTPUT_SCHEMA, messages)
        answer = str(parsed["answer"])
        _record(
            run,
            kind="question",
            prompt=question.text,
            system_prompt=QUESTION_SYSTEM_PROMPT,
            response_text=answer,
            structured_response=parsed,
            raw_response=raw,
            callback=on_turn,
            question=question,
        )
        run.question_results.append(_question_result(question, answer))
        question_history.append(AIMessage(answer))

    final_messages = [
        SystemMessage(FINAL_SYSTEM_PROMPT),
        *question_history,
        HumanMessage(MULTI_FINAL_PROMPT),
    ]
    raw, parsed = _invoke_structured(model, FINAL_OUTPUT_SCHEMA, final_messages)
    final_scenario = str(parsed["final_scenario"])
    final_turn = _record(
        run,
        kind="final",
        prompt=MULTI_FINAL_PROMPT,
        system_prompt=FINAL_SYSTEM_PROMPT,
        response_text=final_scenario,
        structured_response=parsed,
        raw_response=raw,
        callback=on_turn,
    )
    run.final_scenario = final_scenario
    run.final_scenario_word_count = final_turn.scenario_word_count or 0
    run.final_scenario_word_count_valid = bool(final_turn.scenario_word_count_valid)
    run.completed_at = datetime.now(timezone.utc)
    return run
