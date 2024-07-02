import pytest
from modelsmith.language_models import (
    OpenAIModel,
    VertexAIChatModel,
    VertexAIGenerativeModel,
    VertexAITextGenerationModel,
)
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel

MODEL_INSTANCE_PARAMS = [
    pytest.param(
        TextGenerationModel.from_pretrained("text-bison"), id="text_model_old_style"
    ),
    pytest.param(VertexAITextGenerationModel("text-bison"), id="text_model"),
    pytest.param(ChatModel.from_pretrained("chat-bison"), id="chat_model_old_style"),
    pytest.param(VertexAIChatModel("chat-bison"), id="chat_model"),
    pytest.param(VertexAIGenerativeModel("gemini-1.0-pro"), id="gemini_1_0_pro"),
    pytest.param(VertexAIGenerativeModel("gemini-1.5-pro"), id="gemini_1_5_pro"),
    pytest.param(GenerativeModel("gemini-1.5-flash"), id="gemini_1_5_flash_old_style"),
    pytest.param(VertexAIGenerativeModel("gemini-1.5-flash"), id="gemini_1_5_flash"),
    pytest.param(OpenAIModel("gpt-3.5-turbo"), id="gpt-3.5-turbo"),
    pytest.param(OpenAIModel("gpt-4o"), id="gpt-4o"),
]

MODEL_SETTINGS_PARAMS = [
    pytest.param({"temperature": 0.0}, id="temp_zero"),
    pytest.param({"temperature": 0.9}, id="temp_point_nine"),
]
