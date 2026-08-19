"""Prompt templates for the SINGLE condition and each MULTI phase."""

from .questions import ScenarioQuestion
from .text_metrics import MAX_SCENARIO_WORDS, MIN_SCENARIO_WORDS

SINGLE_SYSTEM_PROMPT = """You are completing a household food-waste scenario-construction task.
Develop one household situation using only information introduced in the request. Follow the requested
output structure, keep the question answers concise, and do not provide advice or interventions."""


INITIAL_SYSTEM_PROMPT = """You initialize a household food-waste situation from the supplied attributes.
Choose one concrete focal food that belongs to the specified food category. Establish a plausible
circumstance in which the specified waste reason is the primary explanation for the food being at risk
of waste. Return only the requested structured starting state. Do not describe the complete sequence of
events or its eventual outcome."""


QUESTION_SYSTEM_PROMPT = """You are answering a sequence of questions about one household food-waste
situation that has already been initialized.

Use the established starting information and all previous answers as context. At each turn, answer only
the current question in 1-3 concise sentences. Add only the information needed to answer that question,
and remain consistent with all previously established details.

Do not restart or summarize the situation. Do not anticipate or answer later questions. Do not combine
the accumulated information into a complete account. Do not provide advice or interventions, and do not
discuss the question-answering process."""


FINAL_SYSTEM_PROMPT = f"""You are producing a final household food-waste scenario from an established
starting point and a completed sequence of question-and-answer pairs.

Use only information established in the starting point and previous answers. Combine that information
into one coherent household food-waste scenario between {MIN_SCENARIO_WORDS} and {MAX_SCENARIO_WORDS}
words, inclusive. Preserve the established focal food, household circumstances, sequence of events, and
primary explanation for the waste. Do not introduce substantive new facts solely to make the scenario
more complete. Do not mention the questions, answers, prompts, or construction process."""


def target_block(food_category: str, waste_reason: str) -> str:
    return (
        "The starting situation has two sampled attributes:\n"
        f"Food category: {food_category}\n"
        f"Primary reason for waste: {waste_reason}"
    )


def multi_initial_prompt(food_category: str, waste_reason: str) -> str:
    return f"""We will progressively construct one household food-waste situation.

{target_block(food_category, waste_reason)}

Establish the starting point in 1-2 sentences. Choose one concrete focal food belonging to the
specified category and establish a plausible circumstance in which the specified reason is the
primary explanation for its risk of being wasted. Do not yet describe the complete sequence of
events or the final outcome."""


def multi_question_handoff(focal_food: str, starting_point: str) -> str:
    return f"""The initialization phase established the following starting information:

Focal food: {focal_food}
Starting point: {starting_point}

Use this information as the starting context for the questions that follow."""


MULTI_FINAL_PROMPT = "Generate the final household food-waste scenario now."


def single_prompt(
    food_category: str,
    waste_reason: str,
    questions: list[ScenarioQuestion],
) -> str:
    numbered = "\n".join(f"{question.id}. {question.text}" for question in questions)
    return f"""We will construct one household food-waste situation in a single turn.

{target_block(food_category, waste_reason)}

First, briefly establish the starting point by choosing one concrete focal food belonging to the
specified category and establishing a plausible circumstance in which the specified reason is the
primary explanation for its risk of being wasted.

Then answer each question briefly. The answers must describe one continuous household situation,
not independent examples.

{numbered}

Finally, write one coherent household food-waste scenario between {MIN_SCENARIO_WORDS} and
{MAX_SCENARIO_WORDS} words, inclusive, using the information established in the answers. Do not
introduce substantive new facts solely to complete the story. Return one answer for every supplied
question ID, keyed by that ID as required by the response schema."""
