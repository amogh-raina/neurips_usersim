"""Create one standalone OpenRouter chat model; no agents or tools are involved."""

import os

from langchain_openrouter import ChatOpenRouter
from openrouter.utils import BackoffStrategy, RetryConfig


def _automatic_reasoning_effort(model_name: str) -> str:
    """Use the lowest documented effort for the selected reasoning family."""
    short_name = model_name.rsplit("/", 1)[-1]
    if short_name == "gpt-5" or short_name.startswith(("gpt-5-mini", "gpt-5-nano")):
        return "minimal"
    return "low"


def create_chat_model(model_name: str):
    timeout_seconds = float(os.getenv("USERSIM_REQUEST_TIMEOUT", "120"))
    model = ChatOpenRouter(
        model=model_name,
        # langchain-openrouter maps this value to the SDK's timeout_ms field.
        timeout=round(timeout_seconds * 1000),
        max_retries=0,
    )
    # langchain-openrouter 0.2.8 leaves the SDK retry setting unset when
    # max_retries=0. The SDK then applies its own one-hour default retry window.
    # Set an explicit no-retry policy on the already-created SDK client.
    model.client.sdk_configuration.retry_config = RetryConfig(
        strategy="none",
        backoff=BackoffStrategy(
            initial_interval=0,
            max_interval=0,
            exponent=1.0,
            max_elapsed_time=0,
            jitter_ms=0,
        ),
        retry_connection_errors=False,
    )
    configured_effort = os.getenv("USERSIM_REASONING_EFFORT", "auto").strip().lower()
    if configured_effort not in {"auto", "none", "minimal", "low", "medium", "high"}:
        raise ValueError(
            "USERSIM_REASONING_EFFORT must be auto, none, minimal, low, medium, or high."
        )

    # LangChain's OpenRouter profile tells us whether the model is reasoning-capable.
    # Do not send a reasoning parameter to explicitly non-reasoning models.
    if model.profile and model.profile.get("reasoning_output"):
        effort = (
            _automatic_reasoning_effort(model_name)
            if configured_effort == "auto"
            else configured_effort
        )
        model.reasoning = {"effort": effort}
    return model


def effective_reasoning_effort(model: ChatOpenRouter) -> str | None:
    return (model.reasoning or {}).get("effort")
