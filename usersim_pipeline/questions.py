"""Aspect-aware scenario question pool from the project workplan."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioQuestion:
    id: str
    aspect_id: int
    aspect: str
    text: str


ASPECTS = {
    1: "Household configuration, routines & coordination",
    2: "Planning, intended use & inventory awareness",
    3: "Acquisition, purchasing & quantity",
    4: "Storage, organization, visibility & time",
    5: "Preparation, consumption & leftover management",
    6: "Household contingencies, competing demands & changing circumstances",
    7: "Recognition, assessment, available actions & decision",
}

def _question(number: int, aspect_id: int, text: str) -> ScenarioQuestion:
    question_id = f"Q{number}"
    return ScenarioQuestion(
        id=question_id,
        aspect_id=aspect_id,
        aspect=ASPECTS[aspect_id],
        text=text,
    )


QUESTION_POOL = [
    _question(1, 1, "Who lives in the household and is involved in managing food?"),
    _question(2, 1, "How are shopping, cooking and food-management responsibilities usually shared?"),
    _question(3, 1, "What is the household's normal meal routine?"),
    _question(4, 1, "What time or schedule constraints characterize this period?"),
    _question(5, 1, "Do household members have different food needs, appetites or preferences that matter in this situation?"),
    _question(6, 2, "What planning, if any, had the household done for meals during this period?"),
    _question(7, 2, "What did the household know about the food it already had available?"),
    _question(8, 2, "What was the focal item originally expected to be used for?"),
    _question(9, 2, "When was it expected to be used?"),
    _question(10, 3, "How did the focal item enter the household?"),
    _question(11, 3, "Under what circumstances was it obtained?"),
    _question(12, 3, "How much of it was available relative to what the household expected to need?"),
    _question(13, 3, "What other food or meal commitments existed around the same time?"),
    _question(14, 4, "Where was the focal item kept?"),
    _question(15, 4, "How was it stored or handled?"),
    _question(16, 4, "How visible or easy to keep track of was it?"),
    _question(17, 4, "Were any storage-space, packaging or equipment constraints relevant?"),
    _question(18, 4, "How much time passed before the item next became relevant?"),
    _question(19, 5, "Was any of the focal item prepared or served during this period?"),
    _question(20, 5, "How much of it, if any, was actually used or eaten?"),
    _question(21, 5, "If some remained, what happened to it immediately afterward?"),
    _question(22, 5, "What opportunities were there to use it again?"),
    _question(23, 6, "What else was happening in the household while this situation developed?"),
    _question(24, 6, "What other activities or responsibilities demanded attention during this period?"),
    _question(25, 6, "What other meals or food needs were being managed at the same time?"),
    _question(26, 6, "How did household members' schedules affect what they were able to do?"),
    _question(27, 7, "When the item was next considered, what did household members know about its condition and history?"),
    _question(28, 7, "How did they judge whether it could still be used?"),
    _question(29, 7, "What possible actions did they perceive at that point?"),
    _question(30, 7, "What practical constraints affected those options?"),
    _question(31, 7, "What did the household eventually do with the item?"),
]

# Backward-compatible name for callers that previously used a selected subset.
# The experimental instrument is now the complete, fixed, ordered pool.
DEFAULT_QUESTIONS = QUESTION_POOL
