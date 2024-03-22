import pytest
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel


@pytest.fixture
def text_model() -> TextGenerationModel:
    return TextGenerationModel.from_pretrained("text-bison")


@pytest.fixture
def chat_model() -> ChatModel:
    return ChatModel.from_pretrained("chat-bison")


@pytest.fixture
def generative_model() -> GenerativeModel:
    return GenerativeModel("gemini-1.0-pro")
