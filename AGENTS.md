# AGENTS.md

## Project purpose

This repository implements a controlled experiment comparing SINGLE and MULTI LLM generation of household food-waste scenarios. Every case begins with two calibrated targets: a broad food category and a primary reason for waste. The experiment measures whether those targets remain faithful as scenario detail is introduced.

The source workplan is the Google Doc `usersimWorkshopWorkplan`:
https://docs.google.com/document/d/1mT7HjPR_2TSYBhqRNCF3_zT0HmZYug_2l0OtYVqwrDc/edit

## Current scope

- Python 3.14 application.
- Virtual environment directory and prompt: `sim_neurips`.
- Standalone LangChain chat-model calls through `langchain-openrouter`.
- Streamlit UI, per-run CLI, and resumable dataset-batch CLI.
- No agents, tools, retrieval, LangGraph, or autonomous orchestration.
- JSON persistence for complete experiment records.

## Setup

```bash
cd /Users/lzp112/Projects/neurips_sim
python3.14 -m venv --prompt sim_neurips sim_neurips
source sim_neurips/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Required secret: `OPENROUTER_API_KEY` in `.env`. Never place secrets in `.env.example`, source files, tests, logs, or documentation.

Run the UI with `streamlit run app.py`. Run tests with `pytest`.

## Dependency baseline

- Python 3.14.x
- langchain 1.3.15
- langchain-openrouter 0.2.8
- pydantic 2.12.5 is transitive only (latest compatible with the current OpenRouter SDK's `<2.13` requirement)
- python-dotenv 1.2.3
- streamlit 1.61.1
- pytest 9.1.1

Local records use dataclasses and model outputs use JSON Schema rather than application-defined Pydantic models. LangChain Core still imports a Pydantic-v1 compatibility namespace on Python 3.14; `usersim_pipeline.__init__` narrowly suppresses that known upstream warning. Recheck the workaround during upgrades.

## Documentation policy

For LangChain questions, use the LangChain Docs MCP server first when it is available. Otherwise restrict web documentation access to these user-approved official pages:

- https://docs.langchain.com/oss/python/langchain/models
- https://docs.langchain.com/oss/python/langchain/messages
- https://docs.langchain.com/oss/python/langchain/streaming
- https://docs.langchain.com/oss/python/langchain/structured-output
- https://docs.langchain.com/oss/python/langchain/short-term-memory

Do not substitute general web sources unless the user explicitly changes this restriction.

## Architecture

- `app.py`: dynamic Streamlit inputs and run visualization.
- `data/finnish_household_food_waste_target_pairs.csv`: frozen 100-pair experimental input dataset.
- `usersim_pipeline/model_factory.py`: creates `ChatOpenRouter` with no tools.
- `usersim_pipeline/batch_cli.py`: concurrent, resumable execution across frozen target pairs.
- `usersim_pipeline/questions.py`: fixed ordered 31-question pool and seven aspect labels.
- `usersim_pipeline/prompts.py`: SINGLE prompt plus isolated MULTI initialization, question, and final-synthesis prompts.
- `usersim_pipeline/runners.py`: input validation and SINGLE/MULTI execution.
- `usersim_pipeline/schemas.py`: dataclass run records and provider-facing JSON Schemas.
- `usersim_pipeline/text_metrics.py`: deterministic local word counting for final scenarios.
- `usersim_pipeline/storage.py`: saves JSON to `runs/`.
- `usersim_pipeline/target_pairs.py`: validates and loads the frozen target-pair CSV.
- `usersim_pipeline/cli.py`: command-line entry point.
- `tests/test_runners.py`: fake-model tests for call count, phase isolation, history accumulation, final prompts, and word counts.
- `tests/test_model_factory.py`: timeout, retry, and reasoning-configuration tests.
- `tests/test_dataset.py`: deterministic dataset schema, ID, label, flag, and marginal validation.
- `tests/test_batch_cli.py`: batch call-count and resume-identity validation.

`usersim_pipeline.egg-info/` is generated editable-install metadata. Do not edit or commit it.

## Pipeline invariants

Preserve these unless the user explicitly changes the experiment:

1. SINGLE makes exactly one model call.
2. MULTI makes one initial call, one call per question, and one final-synthesis call.
3. MULTI isolates Turn 0 from the question phase. Question calls receive the starting-state handoff and full preceding question/answer history, but not the Turn 0 instructions or sampled target labels.
4. MULTI uses separate initialization, question, and final-synthesis system prompts. The final call replaces the question-stage system prompt while retaining the accumulated handoff and Q/A history.
5. Question turns should not remind the model of the calibrated food category or waste reason.
6. Final synthesis should reuse established information instead of inventing substantive new facts.
7. Preserve exact inputs, prompts, responses, timestamps, settings, and available usage metadata in the JSON record. CLI compact files store each phase system prompt once at top level; reconstruct a turn's system instruction from its `kind` and `system_prompts` rather than duplicating the full text on every turn.
8. Use direct model-level `with_structured_output(..., method="json_schema", strict=True)`; do not create an agent for response formatting.
9. Store aspect metadata for analysis but do not send aspect labels to the model unless the experimental design explicitly changes.
10. Use the complete Q1-Q31 pool exactly once in both conditions. SINGLE uses canonical order. Dataset-batch MULTI uses a deterministic pair-specific permutation derived from the recorded question-order seed and `pair_id`; the UI question table remains read-only.
11. Keep interactive timeout/retry behavior explicit. SINGLE has no completed turn to display until its one structured response returns; do not label that period as preparation.
12. Do not send `temperature`; selected OpenRouter reasoning models may reject it. Record and match the effective reasoning effort instead.
13. `USERSIM_REQUEST_TIMEOUT` is expressed in seconds for users but must be converted to milliseconds for `langchain-openrouter`. Keep the OpenRouter SDK retry strategy explicitly set to `none` while the integration's zero-retry behavior leaves it unset.
14. Final scenarios in both SINGLE and MULTI are instructed to contain 200–350 words, inclusive. Use the shared constants in `text_metrics.py`; do not duplicate numeric limits across code.
15. Final-scenario word counts are computed locally with `count_words()` and must not be generated by the model.
16. Store the deterministic count and validity as `ExperimentRun.final_scenario_word_count` / `final_scenario_word_count_valid` and as `TurnRecord.scenario_word_count` / `scenario_word_count_valid` for SINGLE/final turns. Report out-of-range outputs without automatically rejecting, rewriting, or retrying them unless the experiment explicitly changes.
17. SINGLE `question_answers` must be a strict object keyed by every supplied question ID, with all IDs required and additional properties forbidden. Do not revert to an array of `{question_id, answer}` objects, which permits duplicate IDs and omissions.
18. With the current 31 questions, a complete frozen dataset run is 100 SINGLE calls plus 3,300 MULTI calls: 3,400 total. MULTI has 33 calls per pair (`1 + 31 + 1`), not 34.
19. Batch concurrency is across independent pair/condition trajectories only. Calls inside one MULTI trajectory remain sequential and retain complete prior history.
20. Use one stable OpenRouter `session_id` per pair/condition trajectory. Do not share a session ID between matched SINGLE and MULTI runs.
21. Do not add hidden model retries to batch execution. Record failures and use the deterministic dataset-run folder, pair ID, manifest, and `--resume` behavior to continue safely.
22. Resume must verify the dataset hash, exact selected pair IDs, selection method and seed, question-order seed, model, selected modes, and question count before skipping completed outputs. A completed MULTI file must contain the exact expected pair-specific question order.
23. Never use a response cache for experimental generations. Provider-side prompt-prefix caching is allowed; keep the fixed SINGLE prompt/question material before its dynamic target block unless the design explicitly changes.
24. The current runner does not use OpenRouter's asynchronous Batch API. MULTI depends on prior outputs and would require sequential batch waves; do not describe its 3,300 calls as one independently executable batch.
25. Batch selection supports all rows, first N, or a reproducible random N. Apply one selected pair list to both conditions for `--mode both`, and store it under `runs/dataset_run_<N>/single|multi/<pair_id>.json` by default.
26. Keep the MULTI question-stage system prompt identical across all dataset pairs. Randomize only the ordered Q1-Q31 sequence, without replacement. Preserve that order in the in-memory `questions` plus `turns` and `question_results`; compact CLI files intentionally omit the redundant top-level `questions` copy.
27. `usersim-run` and `usersim-batch` save `cli_compact_v1`: omit the redundant top-level `questions` array, repeated turn-level `system_prompt`/`aspect`, and null-valued turn fields. Treat ordered `question_results` as the authoritative question/aspect/answer table. Keep Streamlit's full serialization available separately.
28. MULTI has one logical question-stage system message at the beginning of its question conversation. Because `ChatOpenRouter.invoke(messages)` is stateless, every question request must still contain that original system message exactly once together with the complete accumulated handoff and Q/A history. Never append duplicate system messages inside the history, and do not assume `session_id` provides server-side conversational memory.
29. A repeated turn-level `system_prompt` value in legacy/full JSON is audit metadata identifying the instruction that governed that call; it is not evidence that another system message was appended to the conversation. Compact CLI JSON stores the mapping once in top-level `system_prompts` and resolves it through `turn.kind`.

## Questions and prompts

The fixed questions come from the workplan and cover household context, planning, acquisition, storage, preparation, competing demands, food assessment, and disposal. Change and version the pool in `questions.py`, not per run in the UI.

Prompt definitions belong in `prompts.py`, not inline in the UI or runners. Preserve the phase boundary: only the final MULTI system prompt should instruct scenario generation, and the MULTI question-stage system prompt must not use the term “scenario.” SINGLE and MULTI continue to share `target_block()`.

## Change guidelines

- Keep model/provider construction isolated in `model_factory.py`.
- Keep UI concerns out of runner logic; use the existing callback to surface turns.
- Keep runners callable without Streamlit so tests and batch scripts can reuse them.
- Do not introduce an agent or tool layer for straightforward model invocation.
- Add or update fake-model tests for any change to call counts, prompts, message ordering, history, or serialization.
- Do not make paid model calls during tests.
- Use `apply_patch` for source edits and preserve unrelated user changes.
- Run `pytest` and a syntax check after implementation changes.
- Restart Streamlit after changing imported dataclasses or pipeline modules; hot reload can retain stale class definitions. Keep UI reads backward-compatible with older in-memory run objects when practical.
- Keep generated run JSON, virtual environments, caches, `.env`, and `*.egg-info/` out of Git.

## Known follow-up work

- Freeze and version the final production question list.
- Add explicit ontology definitions for food categories and waste reasons if required.
- Add coding and drift-analysis pipelines after generation is stable.
