"""Streamlit interface for inspecting SINGLE and MULTI experiment runs."""

from html import escape
import os

import streamlit as st
from dotenv import load_dotenv

from usersim_pipeline.model_factory import create_chat_model, effective_reasoning_effort
from usersim_pipeline.questions import QUESTION_POOL
from usersim_pipeline.runners import run_multi_turn, run_single_turn
from usersim_pipeline.storage import save_run
from usersim_pipeline.text_metrics import (
    MAX_SCENARIO_WORDS,
    MIN_SCENARIO_WORDS,
    count_words,
    scenario_word_count_is_valid,
)

load_dotenv()


def render_grouped_questions(questions, answers_by_id=None):
    """Render each aspect once with its questions as stacked subrows."""
    grouped = {}
    for question in questions:
        grouped.setdefault((question.aspect_id, question.aspect), []).append(question)

    aspect_blocks = []
    for (aspect_id, aspect), aspect_questions in grouped.items():
        question_rows = []
        for question in aspect_questions:
            answer = (answers_by_id or {}).get(question.id)
            answer_html = (
                f'<div class="question-answer">{escape(answer)}</div>'
                if answer is not None
                else ""
            )
            question_rows.append(
                '<div class="question-subrow">'
                f'<span class="question-id">{escape(question.id)}</span>'
                '<div class="question-content">'
                f'<div>{escape(question.text)}</div>{answer_html}'
                '</div></div>'
            )
        aspect_blocks.append(
            '<section class="aspect-block">'
            '<div class="aspect-cell">'
            f'<span class="aspect-number">Aspect {aspect_id}</span>'
            f'<strong>{escape(aspect)}</strong>'
            '</div>'
            f'<div class="question-cell">{"".join(question_rows)}</div>'
            '</section>'
        )

    st.html(
        """
        <style>
          .question-pool { display: grid; gap: .75rem; }
          .aspect-block {
            display: grid;
            grid-template-columns: minmax(13rem, 30%) 1fr;
            border: 1px solid rgba(128, 128, 128, .28);
            border-radius: .65rem;
            overflow: hidden;
          }
          .aspect-cell {
            display: flex;
            flex-direction: column;
            gap: .3rem;
            padding: .9rem 1rem;
            background: rgba(128, 128, 128, .09);
          }
          .aspect-number {
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .04em;
            opacity: .65;
            text-transform: uppercase;
          }
          .question-cell { min-width: 0; }
          .question-subrow {
            display: grid;
            grid-template-columns: 3rem 1fr;
            gap: .7rem;
            padding: .72rem 1rem;
            align-items: start;
          }
          .question-subrow + .question-subrow {
            border-top: 1px solid rgba(128, 128, 128, .2);
          }
          .question-id { font-weight: 700; opacity: .72; }
          .question-content { min-width: 0; }
          .question-answer {
            margin-top: .4rem;
            padding-top: .4rem;
            border-top: 1px dashed rgba(128, 128, 128, .28);
            opacity: .82;
          }
          @media (max-width: 700px) {
            .aspect-block { grid-template-columns: 1fr; }
          }
        </style>
        """
        f'<div class="question-pool">{"".join(aspect_blocks)}</div>'
    )

st.set_page_config(page_title="UserSim Workshop", page_icon="🧪", layout="wide")
st.title("Household food-waste scenario runner")
st.caption("Compare one-shot generation with explicit conversational-history generation.")

with st.sidebar:
    st.header("Run settings")
    mode = st.radio("Pipeline", ["Single turn", "Multi turn"])
    model_name = st.text_input(
        "OpenRouter model",
        value=os.getenv("USERSIM_MODEL", "openai/gpt-4.1-mini"),
        help="Use an OpenRouter model identifier, for example openai/gpt-4.1-mini.",
    )
    request_timeout = float(os.getenv("USERSIM_REQUEST_TIMEOUT", "120"))
    max_retries = 0
    reasoning_policy = os.getenv("USERSIM_REASONING_EFFORT", "auto")
    st.caption(
        f"Reasoning: {reasoning_policy} · Request timeout: {request_timeout:g}s · "
        f"Automatic retries: {max_retries}"
    )

left, right = st.columns(2)
with left:
    food_category = st.text_input("Food category", placeholder="e.g. vegetables")
with right:
    waste_reason = st.text_input("Primary waste reason", placeholder="e.g. changed plans")

questions = QUESTION_POOL
with st.expander("Scenario question pool (fixed for every run)", expanded=False):
    st.caption(
        "The aspects are question metadata displayed and saved for analysis. Only the ordered "
        "question text is sent to the model, and all 31 questions are used in both pipelines."
    )
    render_grouped_questions(questions)
estimated_calls = 1 if mode == "Single turn" else len(questions) + 2
st.info(f"This run will make {estimated_calls} model call{'s' if estimated_calls != 1 else ''}.")
if mode == "Single turn":
    st.caption(
        "SINGLE returns one complete structured response, so no intermediate answers "
        "can be displayed while that request is running."
    )

if st.button("Run scenario", type="primary", width="stretch"):
    if not food_category.strip() or not waste_reason.strip():
        st.error("Enter both a food category and a primary waste reason.")
    else:
        rendered_turns = []
        waiting_text = (
            f"Waiting for the single structured response (timeout: {request_timeout:g}s)…"
            if mode == "Single turn"
            else "Requesting the initial scenario state…"
        )
        progress = st.progress(0.05, text=waiting_text)
        output = st.container()

        def show_turn(turn, run):
            rendered_turns.append(turn)
            progress.progress(
                len(rendered_turns) / estimated_calls,
                text=f"Completed {len(rendered_turns)} of {estimated_calls} calls",
            )
            with output:
                label = {
                    "single": "Single-turn output",
                    "initial": "Turn 0 — initial state",
                    "question": f"Turn {turn.turn_index} — question",
                    "final": "Final synthesis",
                }[turn.kind]
                with st.expander(label, expanded=turn.kind in {"single", "final"}):
                    if turn.question_id:
                        st.caption(f"{turn.question_id} · {turn.aspect}")
                    st.markdown(f"**Input**\n\n{turn.prompt}")
                    st.markdown(f"**Model response**\n\n{turn.response}")
                    scenario_word_count = getattr(turn, "scenario_word_count", None)
                    if scenario_word_count is None and turn.kind in {"single", "final"}:
                        scenario_word_count = count_words(turn.response)
                    if scenario_word_count is not None:
                        word_count_valid = getattr(
                            turn,
                            "scenario_word_count_valid",
                            scenario_word_count_is_valid(scenario_word_count),
                        )
                        word_count_message = (
                            f"Scenario word count: {scenario_word_count} "
                            f"(required range: {MIN_SCENARIO_WORDS}–{MAX_SCENARIO_WORDS})"
                        )
                        if word_count_valid:
                            st.caption(word_count_message)
                        else:
                            st.warning(f"{word_count_message} — outside the required range")
                    st.markdown("**Structured response**")
                    st.json(turn.structured_response)
                    if turn.usage_metadata:
                        st.caption(f"Usage: {turn.usage_metadata}")

        try:
            model = create_chat_model(model_name.strip())
            runner = run_single_turn if mode == "Single turn" else run_multi_turn
            with st.spinner(waiting_text, show_time=True, width="stretch"):
                run = runner(
                    model,
                    food_category=food_category,
                    waste_reason=waste_reason,
                    questions=questions,
                    model_name=model_name.strip(),
                    reasoning_effort=effective_reasoning_effort(model),
                    on_turn=show_turn,
                )
            saved_path = save_run(run)
            st.session_state["last_run"] = run
            progress.progress(1.0, text="Run complete")
            st.success(f"Completed and saved locally as {saved_path.name}")
        except Exception as exc:
            progress.empty()
            if "timeout" in f"{type(exc).__name__} {exc}".lower():
                st.error(
                    f"The model did not complete within {request_timeout:g} seconds. "
                    "No automatic retry was started. Try again or select a faster OpenRouter model."
                )
            st.exception(exc)

if run := st.session_state.get("last_run"):
    st.divider()
    st.subheader("Final scenario")
    st.write(run.final_scenario)
    final_scenario_word_count = getattr(
        run,
        "final_scenario_word_count",
        count_words(run.final_scenario),
    )
    final_word_count_valid = getattr(
        run,
        "final_scenario_word_count_valid",
        scenario_word_count_is_valid(final_scenario_word_count),
    )
    final_word_count_message = (
        f"Word count: {final_scenario_word_count} "
        f"(required range: {MIN_SCENARIO_WORDS}–{MAX_SCENARIO_WORDS})"
    )
    if final_word_count_valid:
        st.caption(final_word_count_message)
    else:
        st.warning(f"{final_word_count_message} — outside the required range")
    with st.expander("Question answers and aspect metadata", expanded=False):
        render_grouped_questions(
            run.questions,
            answers_by_id={result.question_id: result.answer for result in run.question_results},
        )
    st.download_button(
        "Download complete run JSON",
        data=run.to_json(indent=2),
        file_name=f"{run.run_id}.json",
        mime="application/json",
        width="stretch",
    )
