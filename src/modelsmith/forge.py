import logging
from typing import Any, Generic, Iterable, TypeVar

from pydantic import ValidationError
from tenacity import RetryError, Retrying, stop_after_attempt
from vertexai.generative_models import GenerationResponse, GenerativeModel
from vertexai.language_models import (
    ChatModel,
    TextGenerationModel,
    TextGenerationResponse,
)

from modelsmith.enums import ResponseModelType
from modelsmith.exceptions import (
    CombinedException,
    ModelNotDerivedError,
    PatternNotFound,
)
from modelsmith.language_model import LanguageModelWrapper
from modelsmith.prompt import Prompt
from modelsmith.response_model import ResponseModel
from modelsmith.utilities import find_patterns

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Forge(Generic[T]):
    """
    Class used to extract a Pydantic model or Python class instance from the response
    of an LLM.
    """

    def __init__(
        self,
        *,
        model: ChatModel | GenerativeModel | TextGenerationModel,
        response_model: type[T],
        prompt: str | None = None,
        match_patterns: str | Iterable[str] = (r"```json(.*?)```", r"\{.*\}"),
        max_retries: int = 3,
        raise_on_failure: bool = True,
    ) -> None:
        """
        :param model: The LLM model to use for the Forge.
        :param response_model: The Pydantic model or Python type to return from the LLM.
                               e.g. list[str] or a Pydantic model class.
        :param prompt: The prompt to use for the Forge. This can be jinja2 templated
                       text with placeholders replaced with keyword arguments passed
                       to the Forge's 'generate_response()' method.
        :param match_patterns: A list of regex patterns to match against the LLM
                               response. The first match will be used to extract the
                               object that will be passed to the Pydantic model set
                               as the response_model.
        :param max_retries: The maximum number of times to retry getting the required
                            structured response from the LLM.
        :param raise_on_failure: If True, raise an error if the Forge fails to
                                 return a response in the form of the response_model. If
                                 False, return None.
        """
        self.model = LanguageModelWrapper(model)
        self.response_model = ResponseModel(response_model)
        self.prompt = Prompt(prompt)

        # if a single string is passed add it to a tuple
        self.match_patterns = (
            (match_patterns,) if isinstance(match_patterns, str) else match_patterns
        )

        self.max_retries = max_retries
        self.raise_on_failure = raise_on_failure

    def generate(
        self,
        user_input: str,
        *,
        prompt_values: dict[str, Any] | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> T | None:
        """
        Generate a response from the LLM.

        :param user_input: The user input to pass to the LLM.
        :param prompt_values: The keyword arguments to pass to the prompt template.
        :param model_settings: The keyword arguments to pass to the LLM.
        :return: The response from the LLM parsed into a Pydantic model. If set to not
                 raise an exception on failure, return None if parsing fails.
        """
        prepared_prompt = self.prompt.render(
            user_input=user_input,
            response_model=self.response_model.pydantic_model,
            **(prompt_values or {}),  # pass empty dict to unpack if not prompt_values
        )

        model_response = None
        try:
            for attempt in Retrying(stop=stop_after_attempt(self.max_retries)):
                with attempt:
                    try:
                        response = self.model.send(prepared_prompt, model_settings)
                        model_response = self._process_response(response)
                    except (CombinedException, PatternNotFound) as e:
                        prepared_prompt = (
                            prepared_prompt
                            + f"\nTry again and fix the errors that occurred: {e.args}"
                        )
                        raise e

        except RetryError as e:
            # Raise the final exception if raise_on_failure is True
            if self.raise_on_failure:
                raise ModelNotDerivedError(prepared_prompt) from e

        return model_response

    def _process_response(
        self, llm_response: TextGenerationResponse | GenerationResponse
    ) -> T | None:
        """
        Process the response returned by the LLM into the response_model requested or
        if no response_model was requested return the response as is.

        :param response: Response received from the LLM.
        :return: The pydantic model of the response_model or None.
        """
        logger.debug(f"Processing LLM Response:\n{llm_response.text}")

        model: T | None = None

        json_strings = find_patterns(
            input_string=llm_response.text, patterns=self.match_patterns
        )

        # Check JSON output was found
        if not json_strings:
            raise PatternNotFound("No JSON output found.")

        # Loop through JSON strings found and try to match them to the response
        # model. Return an instance of the response model if it was successfully
        # validated.
        exceptions = []
        for json_string in json_strings:
            try:
                logger.debug(f"Processing JSON String:\n{json_string}")
                model = self.response_model.pydantic_model.model_validate_json(
                    json_string
                )  # type: ignore

                # If the response model is a python type then return the value field
                # of the pydantic model which should contain the python object instance
                if self.response_model.original_type == ResponseModelType.PYTHON:
                    model = model.value  # type: ignore

                # Use the first json match that converts to a model
                break
            except ValidationError as e:
                exceptions.append(e)

        # Raise any errors that occurred during the JSON validation if a valid model
        # was not returned
        if exceptions and model is None:
            raise CombinedException(exceptions)

        return model
