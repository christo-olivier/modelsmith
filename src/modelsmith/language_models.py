import copy
from abc import ABC, abstractmethod
from typing import Any

from anthropic import Anthropic
from openai import OpenAI
from vertexai.generative_models import GenerationResponse, GenerativeModel
from vertexai.language_models import (
    ChatModel,
    TextGenerationModel,
    TextGenerationResponse,
)

DEFAULT_MAX_TOKENS = 1024


class BaseLanguageModel(ABC):
    @abstractmethod
    def send(self, input: str, model_settings: dict[str, Any] | None = None) -> str:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        """
        pass


class AnthropicModel(BaseLanguageModel):
    """
    Class that wraps the Anthropic API to handle sending inputs and receiving outputs.
    """

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        """
        Create a new synchronous anthropic client instance.

        This automatically infers the following arguments from their corresponding
        environment variables if they are not provided:
        - `api_key` from ANTHROPIC_API_KEY
        """
        self.model_name = model_name
        self._client = Anthropic(api_key=api_key)

    def send(self, input: str, model_settings: dict[str, Any] | None = None) -> str:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the text response from the LLM.

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        :return: The response from the LLM.
        """
        # If `max_tokens` not provided in model settings then set it to the default
        # of 1024. Do a deep copy so that the original model settings are not modified.
        settings = copy.deepcopy(model_settings) if model_settings else {}
        if "max_tokens" not in settings:
            settings["max_tokens"] = DEFAULT_MAX_TOKENS

        response = self._client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input}],
            **settings,
        )

        return response.content[0].text


class OpenAIModel(BaseLanguageModel):
    """
    Class that wraps the OpenAI API to handle sending inputs and receiving outputs.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        """
        Construct a new synchronous openai client instance.

        Automatically infers the following arguments from their corresponding
        environment variables if they are not provided:
        - `api_key` from `OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        """
        self.model_name = model_name
        self._client = OpenAI(
            api_key=api_key, organization=organization, project=project
        )

    def send(self, input: str, model_settings: dict[str, Any] | None = None) -> str:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        :return: The response from the LLM.
        """
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input}],
            **(model_settings or {}),
        )  # type: ignore
        return response.choices[0].message.content


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

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        :return: The response from the LLM.
        """
        response = self.chat_session.send_message(input, **(model_settings or {}))
        return response.text


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

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        :return: The response from the LLM.
        """
        response = self.model.generate_content(input, generation_config=model_settings)
        return response.text


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

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        :return: The response from the LLM.
        """
        response = self.model.predict(input, **(model_settings or {}))
        return response.text
