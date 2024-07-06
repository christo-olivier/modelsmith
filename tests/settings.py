import pytest
from modelsmith.language_models import (
    AnthropicModel,
    OpenAIModel,
    VertexAIChatModel,
    VertexAIGenerativeModel,
    VertexAITextGenerationModel,
)

MODEL_INSTANCE_PARAMS = [
    pytest.param(
        AnthropicModel("claude-3-haiku-20240307"), id="anthropic_claude_3_haiku_model"
    ),
    pytest.param(
        AnthropicModel("claude-3-opus-20240229"), id="anthropic_claude_3_opus_model"
    ),
    pytest.param(
        AnthropicModel("claude-3-5-sonnet-20240620"),
        id="anthropic_claude_3.5_sonnet_model",
    ),
    pytest.param(
        VertexAITextGenerationModel("text-bison"), id="vertexai_text_bison_model"
    ),
    pytest.param(VertexAIChatModel("chat-bison"), id="vertexai_chat_bison_model"),
    pytest.param(
        VertexAIGenerativeModel("gemini-1.0-pro"), id="vertexai_gemini_1_0_pro"
    ),
    pytest.param(
        VertexAIGenerativeModel("gemini-1.5-pro"), id="vertexai_gemini_1_5_pro"
    ),
    pytest.param(
        VertexAIGenerativeModel("gemini-1.5-flash"), id="vertexai_gemini_1_5_flash"
    ),
    pytest.param(OpenAIModel("gpt-3.5-turbo"), id="openai_gpt-3.5-turbo"),
    pytest.param(OpenAIModel("gpt-4o"), id="openai_gpt-4o"),
]

MODEL_SETTINGS_PARAMS = [
    pytest.param({"temperature": 0.0}, id="temp_zero"),
    pytest.param({"temperature": 0.9}, id="temp_point_nine"),
]
