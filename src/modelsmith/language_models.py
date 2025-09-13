import copy
from abc import ABC, abstractmethod
from typing import Any

from anthropic import Anthropic
from google import genai
from google.genai.types import GenerateContentConfigDict, HttpOptions
from openai import OpenAI

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
        response = self._client.responses.create(
            model=self.model_name,
            input=input,
            **(model_settings or {}),
        )
        return response.output_text


class GeminiModel(BaseLanguageModel):
    """
    Class that wraps the Google Gemini models to handle sending inputs and
    receiving outputs.
    """

    def __init__(
        self,
        model_name: str,
        vertexai: bool | None = None,
        api_key: str | None = None,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        """
        Create a new Google Gemini model instance.

        Automatically infers the following arguments from their corresponding
            environment variables if they are not provided:
            - `vertexai` from `GOOGLE_GENAI_USE_VERTEXAI`
            - `api_key` from `GOOGLE_API_KEY`
            - `project` from `GOOGLE_CLOUD_PROJECT`
            - `location` from `GOOGLE_CLOUD_LOCATION`
        """
        self._client = genai.Client(
            vertexai=vertexai,
            api_key=api_key,
            project=project,
            location=location,
            http_options=HttpOptions(api_version="v1"),
        )
        self.model_name = model_name

    def send(self, input: str, model_settings: dict[str, Any] | None = None) -> str:
        """
        Send the input to the LLM using the correct method from the underlying model.
        Return the response from the LLM.

        :param input: The input string to send to the LLM.
        :param model_settings: The dictionary containing the model's settings.
        :return: The response from the LLM.
        """
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=input,
            config=GenerateContentConfigDict(**(model_settings or {})),  # type: ignore
        )
        return response.text or ""
