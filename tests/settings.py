import pytest

from modelsmith.language_models import (
    AnthropicModel,
    GeminiModel,
    OpenAIModel,
)

MODEL_INSTANCE_PARAMS = [
    pytest.param(
        AnthropicModel("claude-opus-4-1-20250805"), id="anthropic_claude_4_1_opus"
    ),
    pytest.param(
        AnthropicModel("claude-sonnet-4-20250514"),
        id="anthropic_claude_4_sonnet",
    ),
    pytest.param(
        AnthropicModel("claude-3-5-haiku-20241022"),
        id="anthropic_claude_3_5_haiku",
    ),
    pytest.param(GeminiModel("gemini-2.5-pro"), id="gemini_2_5_pro"),
    pytest.param(GeminiModel("gemini-2.5-flash"), id="gemini_2_5_flash"),
    pytest.param(GeminiModel("gemini-2.5-flash-lite"), id="gemini_2_5_flash_lite"),
    pytest.param(OpenAIModel("gpt-4o"), id="openai_gpt_4o"),
    pytest.param(OpenAIModel("gpt-4.1"), id="openai_gpt_4_1"),
    pytest.param(OpenAIModel("gpt-5"), id="openai_gpt_5"),
]

MODEL_SETTINGS_PARAMS = [
    pytest.param({}, id="model_defaults_settings")
    # pytest.param({"temperature": 0.0}, id="temp_zero"),
    # pytest.param({"temperature": 0.9}, id="temp_point_nine"),
]
