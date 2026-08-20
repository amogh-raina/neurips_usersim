from usersim_pipeline.model_factory import create_chat_model, effective_reasoning_effort


def test_gpt5_nano_omits_temperature_and_uses_minimal(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("USERSIM_REASONING_EFFORT", "auto")

    model = create_chat_model(
        "openai/gpt-5-nano",
        session_id="usersim-test-single-PAIR_001",
        timeout_seconds=45,
    )

    assert "temperature" not in model._default_params
    assert effective_reasoning_effort(model) == "minimal"
    assert model.request_timeout == 45_000
    assert model.session_id == "usersim-test-single-PAIR_001"
    assert model.client.sdk_configuration.retry_config.strategy == "none"


def test_non_reasoning_model_receives_no_reasoning_parameter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("USERSIM_REASONING_EFFORT", "auto")

    model = create_chat_model("openai/gpt-4.1-mini")

    assert "temperature" not in model._default_params
    assert "reasoning" not in model._default_params
    assert effective_reasoning_effort(model) is None
