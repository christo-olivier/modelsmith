import pytest
from modelsmith.language_model import (
    VertexAIChatModel,
    VertexAIGenerativeModel,
    VertexAITextGenerationModel,
)

MODEL_INSTANCE_PARAMS = [
    pytest.param(VertexAITextGenerationModel("text-bison"), id="text_model"),
    pytest.param(VertexAIChatModel("chat-bison"), id="chat_model"),
    pytest.param(VertexAIGenerativeModel("gemini-1.0-pro"), id="gemini_1_0_pro"),
    pytest.param(VertexAIGenerativeModel("gemini-1.5-pro"), id="gemini_1_5_pro"),
    pytest.param(VertexAIGenerativeModel("gemini-1.5-flash"), id="gemini_1_5_flash"),
]

MODEL_SETTINGS_PARAMS = [
    pytest.param({"temperature": 0.0}, id="temp_zero"),
    pytest.param({"temperature": 0.9}, id="temp_point_nine"),
]
