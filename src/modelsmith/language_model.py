import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable

from vertexai.generative_models import GenerationResponse, GenerativeModel
from vertexai.language_models import (
    ChatModel,
    TextGenerationModel,
    TextGenerationResponse,
)

logger = logging.getLogger(__name__)


class BaseLanguageModel(ABC):
    @abstractmethod
    def send(self, input: str, model_settings: dict[str, Any] | None = None) -> Any:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        """
        pass


class OpenAIModel(BaseLanguageModel):
    """
    Class that wraps the OpenAI API to handle sending inputs and receiving outputs.
    """

    pass


class VertexAIChatModel(BaseLanguageModel):
    """
    Class that wraps the Vertex AI ChatModel to handle sending inputs and
    receiving outputs.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model = ChatModel.from_pretrained(model_name)
        self.chat_session = self.model.start_chat()

    def send(
        self, input: str, model_settings: dict[str, Any] | None = None
    ) -> TextGenerationResponse:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.
        """
        return self.chat_session.send_message(input, **(model_settings or {}))


class VertexAIGenerativeModel(BaseLanguageModel):
    """
    Class that wraps the Vertex AI GenerativeModel to handle sending inputs and
    receiving outputs.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model = GenerativeModel(model_name)

    def send(
        self, input: str, model_settings: dict[str, Any] | None = None
    ) -> GenerationResponse:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.
        """
        return self.model.generate_content(input, generation_config=model_settings)


class VertexAITextGenerationModel(BaseLanguageModel):
    """
    Class that wraps the Vertex AI TextGenerationModel to handle sending inputs and
    receiving outputs.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model = TextGenerationModel.from_pretrained(model_name)

    def send(
        self, input: str, model_settings: dict[str, Any] | None = None
    ) -> TextGenerationResponse:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.
        """
        return self.model.predict(input, **(model_settings or {}))


class LanguageModelWrapper:
    """
    Class that wraps the LLM model to handle sending inputs and receiving outputs.
    """

    def __init__(
        self, model: ChatModel | GenerativeModel | TextGenerationModel
    ) -> None:
        self.model = model
        self._llm_send_method = self._init_llm_send_method()

    def _init_llm_send_method(self) -> Callable:
        """
        Set the method to use to send the user input to the LLM.
        """
        warnings.warn(
            (
                "Using VertexAI classes directly is deprecated and will be removed in "
                "a future release. Please import the classes from the `language_model` "
                "module instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(self.model, TextGenerationModel):
            return self.model.predict

        if isinstance(self.model, GenerativeModel):
            return self.model.generate_content

        if isinstance(self.model, ChatModel):
            chat_session = self.model.start_chat()
            return chat_session.send_message

        raise TypeError(
            "The model type must be ChatModel, TextGenerationModel or GenerativeModel"
        )

    def send(
        self, input: str, model_settings: dict[str, Any] | None = None
    ) -> GenerationResponse | TextGenerationResponse:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        """
        # For a GenerativeModel the function signature differs from the other models
        if isinstance(self.model, GenerativeModel):
            return self._llm_send_method(input, generation_config=model_settings)

        return self._llm_send_method(input, **(model_settings or {}))
