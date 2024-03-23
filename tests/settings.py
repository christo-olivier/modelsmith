import pytest
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel

MODEL_INSTANCE_PARAMS = [
    pytest.param(TextGenerationModel.from_pretrained("text-bison"), id="text_model"),
    pytest.param(ChatModel.from_pretrained("chat-bison"), id="chat_model"),
    pytest.param(GenerativeModel("gemini-1.0-pro"), id="generative_model"),
]

MODEL_SETTINGS_PARAMS = [
    pytest.param({"temperature": 0.0}, id="temp_zero"),
    pytest.param({"temperature": 0.9}, id="temp_point_nine"),
]
