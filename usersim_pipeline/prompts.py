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


QUESTION_SYSTEM_PROMPT = """You are a household member participating in a structured interview about
one specific household food situation.

Adopt the perspective and facts established in the initial case description. Treat that description
and all previous answers in this interview as authoritative context.

Answer only the current question in 1-3 concise sentences. Give a direct, concrete, and plausible answer
about this household. If the requested detail has not previously been established, choose one plausible
detail that is consistent with everything already established. Once introduced, treat that detail as a
fact in all subsequent answers.

If the appropriate answer for this household is that something was absent or irrelevant, state that
directly and concretely. Do not say that information is missing, unspecified, unknown, or not provided.

Maintain consistency with the initial information and all previous answers. Do not contradict, replace,
or reinterpret established facts. Do not ask clarifying or follow-up questions. Do not use greetings,
acknowledgements, conversational filler, advice, recommendations, or explanations of your answering
process. Do not summarize the complete situation or anticipate later questions."""


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
    return f"""The interview begins from the following established facts:

Focal food: {focal_food}
Starting information: {starting_point}

Adopt the perspective of a household member involved in this situation. Treat these details and every
answer you subsequently provide as established facts for the remainder of the interview."""


MULTI_FINAL_PROMPT = "Generate the final household food-waste scenario now."


def single_prompt(
    food_category: str,
    waste_reason: str,
    questions: list[ScenarioQuestion],
) -> str:
    numbered = "\n".join(f"{question.id}. {question.text}" for question in questions)
    return f"""We will construct one household food-waste situation in a single turn.

First, briefly establish the starting point by choosing one concrete focal food belonging to the
specified category and establishing a plausible circumstance in which the specified reason is the
primary explanation for its risk of being wasted.

Then answer each question briefly. The answers must describe one continuous household situation,
not independent examples.

{numbered}

Finally, write one coherent household food-waste scenario between {MIN_SCENARIO_WORDS} and
{MAX_SCENARIO_WORDS} words, inclusive, using the information established in the answers. Do not
introduce substantive new facts solely to complete the story. Return one answer for every supplied
question ID, keyed by that ID as required by the response schema.

Apply all of the instructions above to these sampled attributes:

{target_block(food_category, waste_reason)}"""
