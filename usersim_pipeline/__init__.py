"""Single-turn and multi-turn user-simulation experiment pipelines."""

import warnings

# LangChain Core imports the legacy Pydantic-v1 namespace for backward
# compatibility. Pydantic warns about that namespace on Python 3.14 even though
# this project uses JSON Schema and dataclasses rather than Pydantic models.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
    module=r"langchain_core\.utils\.pydantic",
)

from .runners import run_multi_turn, run_single_turn
from .schemas import ExperimentRun, TurnRecord

__all__ = ["ExperimentRun", "TurnRecord", "run_multi_turn", "run_single_turn"]
