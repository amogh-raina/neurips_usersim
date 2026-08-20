# UserSim Workshop pipelines

This project compares whether an LLM preserves two calibrated attributes while constructing a household food-waste scenario:

- the requested **food category**;
- the requested **primary reason for waste**.

It implements two matched generation conditions:

- **SINGLE:** all targets and scenario-building questions are supplied in one model call.
- **MULTI:** Turn 0 initializes a starting state, then isolated question calls receive that handoff and the complete prior question/answer history.

The application uses a standalone LangChain chat model through OpenRouter. It does not use agents, tools, LangGraph, retrieval, or an autonomous loop.

## Codebase structure

```text
neurips_sim/
├── app.py                              # Streamlit UI
├── pyproject.toml                      # Python and dependency configuration
├── .env.example                       # Environment-variable template
├── data/
│   └── finnish_household_food_waste_target_pairs.csv  # Frozen 100-pair input dataset
├── usersim_pipeline/
│   ├── cli.py                          # Command-line interface
│   ├── batch_cli.py                    # Resumable concurrent dataset runner
│   ├── model_factory.py                # ChatOpenRouter construction
│   ├── prompts.py                      # SINGLE and phase-specific MULTI prompts
│   ├── questions.py                    # Full 31-question pool and aspect metadata
│   ├── runners.py                      # SINGLE and MULTI execution logic
│   ├── schemas.py                      # Dataclass records and provider JSON Schemas
│   ├── storage.py                      # JSON persistence
│   ├── target_pairs.py                 # Frozen CSV loader and validation
│   └── text_metrics.py                 # Deterministic local word counting
└── tests/
    ├── test_dataset.py                 # Dataset schema and marginal checks
    ├── test_batch_cli.py               # Batch plan and resume-safety tests
    ├── test_model_factory.py           # Model configuration tests
    └── test_runners.py                 # Pipeline and history tests
```

`usersim_pipeline.egg-info/` is generated packaging metadata created by the editable installation. It is ignored by Git and should not be edited.

## Environment and dependencies

The local virtual environment is named `sim_neurips` and uses Python 3.14.

```bash
cd /Users/lzp112/Projects/neurips_sim
python3.14 -m venv --prompt sim_neurips sim_neurips
source sim_neurips/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Direct dependencies are pinned in `pyproject.toml`:

| Dependency | Version | Purpose |
|---|---:|---|
| Python | 3.14.x | Runtime |
| LangChain | 1.3.15 | Standard message and chat-model interfaces |
| langchain-openrouter | 0.2.8 | `ChatOpenRouter` integration |
| python-dotenv | 1.2.3 | Local `.env` loading |
| Streamlit | 1.61.1 | Interactive run viewer |
| pytest | 9.1.1 | Development tests |

The project does not directly use Pydantic. Local records use standard-library dataclasses and model outputs use JSON Schema. Pydantic 2.12.5 is still installed transitively because the OpenRouter and LangChain packages depend on it.

LangChain Core currently imports its legacy Pydantic-v1 compatibility namespace, which triggers a Python 3.14 warning even when application code does not define Pydantic models. The package initializer narrowly suppresses that known warning for this application; dependency upgrades should recheck whether the workaround is still necessary.

## Configuration

Create `.env` from `.env.example` and add your OpenRouter key:

```dotenv
OPENROUTER_API_KEY=your-key-here
USERSIM_MODEL=openai/gpt-5-nano
USERSIM_REASONING_EFFORT=auto
USERSIM_REQUEST_TIMEOUT=120
```

Never place a real key in `.env.example` or commit `.env`. The latter is ignored by Git.

`model_factory.py` creates one standalone model. The user-facing timeout is expressed in seconds and
converted to the milliseconds expected by `langchain-openrouter`:

```python
timeout_seconds = float(os.getenv("USERSIM_REQUEST_TIMEOUT", "120"))
ChatOpenRouter(
    model=model_name,
    timeout=round(timeout_seconds * 1000),
    max_retries=0,
)
```

The factory also sets the underlying OpenRouter SDK retry strategy explicitly to `none`; otherwise the
SDK can apply its own long retry window even when the LangChain integration receives `max_retries=0`.

Temperature is intentionally omitted rather than set to `None`, making the request compatible with reasoning models such as `openai/gpt-5-nano` that reject temperature changes. The same model and effective reasoning effort are used in either experimental condition.

With `USERSIM_REASONING_EFFORT=auto`, the factory uses `minimal` for the original GPT-5, GPT-5 mini, and GPT-5 nano family; it uses `low` for other models whose LangChain OpenRouter profile marks them as reasoning-capable; and it sends no reasoning parameter to models explicitly marked as non-reasoning. Set the environment value to `none`, `minimal`, `low`, `medium`, or `high` to override the automatic effort for a reasoning model.

## Dynamic inputs

The two required experimental inputs are not hard-coded:

1. `food_category`
2. `waste_reason`

They can be supplied through either interface.

### Streamlit UI

`app.py` provides:

- a SINGLE/MULTI selector;
- food-category and waste-reason text inputs;
- OpenRouter model control and visible reasoning/timeout policy;
- a read-only grouped view showing each aspect once with its question subrows;
- estimated model-call count;
- progress and per-turn output;
- deterministic final-scenario word counts with the required 200–350 range;
- complete JSON download.

The UI validates that both targets are present. It then selects `run_single_turn` or `run_multi_turn` and passes the complete fixed question pool to that runner.

### Command line

The CLI accepts the same targets:

```bash
usersim-run \
  --mode single \
  --food-category vegetables \
  --waste-reason "changed plans"

usersim-run \
  --mode multi \
  --food-category vegetables \
  --waste-reason "changed plans"
```

The CLI uses the complete frozen question pool from `questions.py`. The model and output directory can also be supplied as arguments.

## Target-pair dataset

`data/finnish_household_food_waste_target_pairs.csv` contains the frozen 100 food-category and
primary-waste-reason targets intended for matched SINGLE and MULTI runs. It preserves the required
food-category and waste-reason marginal counts exactly.

The source paper reports those two distributions separately, not as a numerical joint distribution.
The CSV therefore labels its joint distribution as `synthetic_not_empirically_observed`. Following the
study design decision to avoid difficult or incoherent target combinations, the pairs use a
`plausibility_constrained_pairing` allocation rather than independent random pairing. Seed `20260819`
was used to randomize the final row order before assigning `PAIR_001` through `PAIR_100`.

The source is Silvennoinen et al. (2014), *Food waste volume and composition in Finnish households*,
British Food Journal 116(6), 1058–1068, DOI `10.1108/BFJ-12-2012-0311`. The marginal shares describe
measured avoidable household food waste by weight; they are not household-prevalence or disposal-event
rates.

## Running the target-pair dataset

`usersim-batch` reads the frozen CSV and runs every selected pair without involving Streamlit. Start
with a dry run; it validates the dataset and prints the planned work without requiring an API key or
making a model request:

```bash
usersim-batch --mode both --dry-run
```

With 31 questions, the full plan is:

| Condition | Runs | Calls per run | Total model calls |
|---|---:|---:|---:|
| SINGLE | 100 | 1 | 100 |
| MULTI | 100 | 33 | 3,300 |
| Both | 200 | — | 3,400 |

MULTI uses 33 calls, not 34: one initialization, 31 question calls, and one final synthesis. The
question turns inside one trajectory remain sequential because each answer depends on the earlier
answers. Independent food/reason pairs run concurrently.

The same selection is applied to SINGLE and MULTI when `--mode both` is used:

```bash
# First 10 rows in their frozen CSV order
usersim-batch --mode both --num-pairs 10 --selection first --concurrency 4

# A reproducible random sample of 10 rows
usersim-batch \
  --mode both \
  --num-pairs 10 \
  --selection random \
  --sample-seed 42 \
  --output-dir runs/random_seed_42 \
  --concurrency 4

# Run only one experimental condition on the first 10 rows
usersim-batch \
  --mode single --num-pairs 10 --selection first \
  --output-dir runs/single_only --concurrency 4
usersim-batch \
  --mode multi --num-pairs 10 --selection first \
  --output-dir runs/multi_only --concurrency 4

# Full matched 100-pair experiment (all rows is the default)
usersim-batch --mode both --concurrency 4
```

`--limit` remains an alias for `--num-pairs`. Random sampling uses a local seeded sampler and does not
change the frozen CSV. Reusing the same seed reproduces the same ordered pair IDs. Because every
10-pair run is named `dataset_run_10`, use a different `--output-dir` when retaining multiple distinct
10-pair samples.

To continue after failures or interruption, repeat the same selection command with `--resume`:

```bash
usersim-batch --mode both --num-pairs 10 --selection first --concurrency 4 --resume
```

`--request-timeout` is a per-request limit in seconds; there is no global batch timeout. Increase
`--concurrency` cautiously according to model/provider rate limits. Automatic retries remain disabled,
so a failed request is recorded rather than silently increasing the call count. A failed MULTI
trajectory is rerun from its initialization call on resume; partial trajectories are not resumed from
the middle.

The CLI prints one completion/failure line per pair and condition, including the deterministic scenario
word count, then prints the output folder. It does not send dataset runs to the Streamlit UI.

Outputs are deterministic by selected count, target ID, and condition:

```text
runs/dataset_run_10/manifest.json
runs/dataset_run_10/single/PAIR_001.json
runs/dataset_run_10/multi/PAIR_001.json

runs/dataset_run_100/manifest.json
runs/dataset_run_100/single/PAIR_001.json
runs/dataset_run_100/multi/PAIR_001.json
```

Each JSON filename is its source `pair_id`. The manifest stores the selection method, random seed when
applicable, dataset hash, exact selected pair IDs, model, expected call counts, progress, and failures.
Resume refuses a changed dataset, sample, model, condition selection, or question count. Each run record
also stores `batch_id` (for example, `dataset_run_10`) and `target_pair_id`.

### Prompt caching and OpenRouter Batch API

The runner does not use a LangChain response cache: reusing an old generated answer would invalidate
the 100 independent experimental runs. Provider prompt caching is safe because it reuses computation
for an identical input prefix, not the output. SINGLE therefore places its fixed instructions and
question pool before the dynamic target pair, creating a shared prefix across all 100 calls. Whether a
cache is used, its minimum prefix length, and the reported cache-token fields depend on the selected
OpenRouter model/provider.

Every pair/condition trajectory receives a stable OpenRouter `session_id`. This supports sticky routing,
cache locality, and log grouping where the provider supports them. Within MULTI, each question request
extends the same accumulated message prefix, so later turns are the natural caching opportunity.

OpenRouter's asynchronous Batch API is well suited to the 100 independent SINGLE requests and can offer
lower batch pricing, but it is not used by this LangChain runner. The 3,300 MULTI calls cannot be placed
in one independent batch because each later request needs the previous response. Supporting MULTI via
that API would require 33 sequential batch waves and waiting for each wave to finish before constructing
the next. Concurrent synchronous trajectories give useful progress and immediate per-pair persistence;
a separate native Batch API exporter for SINGLE can be added later if cost is more important than
completion latency.

## Questions

`usersim_pipeline/questions.py` contains the fixed 31-question pool from the project workplan, grouped under seven aspects. The UI displays each aspect once with its questions as stacked subrows; there is no per-run question selection.

The questions are intentionally target-neutral: they add scenario detail without repeating the food category or primary reason for waste. This makes it possible to observe whether the model preserves the original targets without reminders.

Question handling is matched across conditions:

- SINGLE inserts the complete ordered list into one prompt and numbers the questions.
- MULTI asks the same ordered list one question per model call.
- The UI displays the full pool in a grouped read-only view, and both conditions receive all 31 questions in the same order.
- For production experiments, the list and order should be frozen and versioned before generating matched cases.

Aspect labels are displayed and saved with each question turn, but are not sent to the model. This supports later aspect-level drift analysis without adding semantic cues such as “competing demands” to the experimental prompt.

Each saved `question_results` row joins the model answer back to authoritative local metadata: question ID, aspect ID, aspect label, and question text. The model is never asked to reproduce that metadata.

## Prompt design

All prompt construction lives in `usersim_pipeline/prompts.py`.

### Phase-specific system prompts

SINGLE uses `SINGLE_SYSTEM_PROMPT`. MULTI deliberately isolates its three phases with separate
system prompts:

- `INITIAL_SYSTEM_PROMPT` establishes the focal food and starting point at Turn 0;
- `QUESTION_SYSTEM_PROMPT` governs concise question answering without mentioning scenario generation;
- `FINAL_SYSTEM_PROMPT` alone instructs the model to synthesize the final scenario.

The Turn 0 instructions are not replayed during question turns, and the question-stage system prompt
is replaced by the final system prompt for synthesis.

### Shared target block

`target_block()` inserts the two dynamic targets once:

```text
Food category: <dynamic food category>
Primary reason for waste: <dynamic waste reason>
```

### SINGLE prompt

`single_prompt()` combines:

1. an instruction to instantiate one concrete focal food;
2. the complete numbered question list;
3. an instruction to treat the answers as one continuous situation;
4. a final synthesis instruction;
5. the shared dynamic target block at the end, after the cacheable fixed prefix;
6. a provider-enforced JSON Schema for the returned fields.

The runner sends one `SystemMessage` and one `HumanMessage`, then invokes the model exactly once. Provider-level JSON Schema requires `focal_food`, `starting_point`, a `question_answers` object keyed by every supplied question ID, and `final_scenario`. Every question ID is a required object property and additional properties are forbidden, preventing duplicated or omitted IDs. The prompt and field description instruct the model to keep the final scenario between 200 and 350 words.

### MULTI prompts

The MULTI flow has four steps across three model phases:

1. **Turn 0:** `INITIAL_SYSTEM_PROMPT` and `multi_initial_prompt()` supply the two targets and request a concrete focal food and starting circumstance.
2. **Handoff:** only the returned `focal_food` and `starting_point` are formatted as question-stage context; the target labels and Turn 0 instructions are not replayed.
3. **Question turns:** `QUESTION_SYSTEM_PROMPT` is used with the handoff and the accumulated question/answer history.
4. **Final synthesis:** `FINAL_SYSTEM_PROMPT` replaces the question-stage system prompt and `MULTI_FINAL_PROMPT` triggers a 200–350-word synthesis.

Each stage has its own strict JSON Schema: Turn 0 returns `focal_food` and `starting_point`, question turns return `answer`, and the final turn returns `final_scenario`.

No question turn repeats the target food category or waste reason.

## How MULTI history works

The MULTI runner manages history explicitly rather than using an agent or memory abstraction:

```text
SystemMessage(question-stage instructions)
HumanMessage(starting-state handoff)
HumanMessage(question 1)
AIMessage(answer 1)
HumanMessage(question 2)
AIMessage(answer 2)
...
```

Before answering a new question, the model receives the starting-state handoff and every preceding
question/answer pair. The final call replaces the first message with `FINAL_SYSTEM_PROMPT` and appends
the synthesis request. With `N` questions, MULTI makes `N + 2` calls: one initial call, `N` question
calls, and one synthesis call. SINGLE always makes one call.

## How SINGLE works

SINGLE sends one `SystemMessage` and one `HumanMessage`. The human message contains the food category, waste reason, and the complete ordered Q1-Q31 question list. The model must establish one starting point, answer every question consistently within that same situation, and synthesize the final scenario in a single generation.

Its JSON Schema requires `focal_food`, `starting_point`, exactly one answer property for every Q1–Q31 ID, and `final_scenario`. The prompt instructs that final scenario to contain 200–350 words. After parsing, the runner verifies the exact key set, then joins each answer to the locally stored aspect metadata in authoritative question order. This produces the same normalized `question_results` records as MULTI while preserving the experimental distinction: SINGLE gets no sequential conversational history, whereas MULTI does.

## Structured model output

Structured output is applied directly to `ChatOpenRouter`; no agent is created:

```python
model.with_structured_output(
    json_schema,
    method="json_schema",
    strict=True,
    include_raw=True,
)
```

`method="json_schema"` sends a provider `response_format` request. `strict=True` requests exact schema adherence, while `include_raw=True` retains the original LangChain message so token-usage metadata remains available. Parsed dictionaries are used for application logic and saved beside the readable response.

Pydantic and `response_format` solve different problems: Pydantic is one optional way to define or validate a schema in Python; `response_format` is how the provider constrains the model response. This project now defines provider schemas as plain JSON Schema dictionaries and uses dataclasses only for local storage.

The selected OpenRouter model must support JSON Schema structured output. If it does not, the run fails explicitly instead of silently reverting to unstructured text.

Interactive requests default to a 120-second timeout with automatic retries disabled. This prevents one slow SINGLE request from being silently repeated for several minutes. The UI displays the configured limit and an elapsed-time spinner; SINGLE cannot show partial question answers because its experimental condition is one complete structured model response.

`langchain-openrouter` 0.2.8 expresses its timeout in milliseconds, so the factory converts `USERSIM_REQUEST_TIMEOUT` from UI-facing seconds to milliseconds. It also installs an explicit OpenRouter SDK retry strategy of `none`; leaving the SDK retry configuration unset activates the SDK's much longer default backoff window.

## Run records and outputs

Dataclass records in `schemas.py` keep both conditions in the same serializable output structure.

Each `ExperimentRun` records:

- unique run ID;
- mode;
- optional batch ID and frozen target-pair ID;
- food category and waste reason;
- model and effective reasoning effort;
- all phase-specific system prompts, the question-stage handoff, and the question list;
- normalized question results with question and aspect metadata;
- start and completion timestamps;
- all turn records;
- final generated output and its deterministic local word count.

Each `TurnRecord` contains:

- turn index and kind;
- exact system prompt used for the turn;
- exact prompt sent for that turn;
- model response;
- a deterministic word count for SINGLE and final-synthesis responses;
- token-usage metadata when the provider returns it.

Word counts are calculated locally rather than generated by the model. Unicode words are counted with
hyphenated terms and contractions treated as one word. This makes the count reproducible and independent
of provider tokenizers. The saved run-level fields are `final_scenario_word_count` and
`final_scenario_word_count_valid`; SINGLE and final turn records contain `scenario_word_count` and
`scenario_word_count_valid`. The 200–350 requirement is prompt-level, while compliance validation is
deterministic: the runner records and visibly flags an out-of-range response but does not automatically
reject, rewrite, or retry it.

`storage.py` writes the record to `runs/<run-id>.json`. The Streamlit UI also offers the same JSON as a download.

In both conditions, `final_scenario` contains only the parsed final scenario. The complete structured response remains available in each turn's `structured_response` field.

## Running the UI

```bash
cd /Users/lzp112/Projects/neurips_sim
source sim_neurips/bin/activate
streamlit run app.py
```

After changing dataclasses or imported pipeline modules, stop and restart Streamlit rather than relying
only on hot reload. A long-running Streamlit process can retain an older `TurnRecord` or `ExperimentRun`
class in memory. The UI uses local fallback counting for older in-memory objects, but a restart is needed
for newly saved JSON to use the latest record schema.

## Experimental controls

For a valid matched comparison, keep these identical across SINGLE and MULTI:

- target food/reason pair;
- question wording and order;
- target definitions, if later added;
- model identifier;
- effective reasoning effort and other generation settings.

The intended manipulation is whether construction occurs in one call or through the explicitly phased
MULTI process. MULTI's phase-specific prompts are fixed experimental materials and must remain identical
across matched MULTI runs.

## Tests

```bash
source sim_neurips/bin/activate
pytest
```

The tests use a fake recording model and require neither an API key nor network access. They verify that:

- SINGLE makes one call containing both targets and every question;
- MULTI makes the expected number of calls;
- Turn 0 instructions and target labels are absent from question-stage calls;
- each question call contains the starting-state handoff and accumulated prior Q/A pairs;
- final synthesis replaces the question-stage system prompt while retaining accumulated context;
- the 200–350-word instructions appear in both generation conditions;
- SINGLE question answers use required Q1–Q31 object keys, preventing duplicate or omitted IDs;
- deterministic word counting is stable and stored on final outputs;
- model timeout, retry, and reasoning settings behave as configured;
- the frozen 100-pair dataset preserves its schema and target marginals.
