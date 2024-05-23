import inspect
import json
import logging
from typing import Any

from jinja2 import BaseLoader, Environment, StrictUndefined, meta, select_autoescape
from pydantic import BaseModel

logger = logging.getLogger(__name__)

RESPONSE_MODEL_TEXT = inspect.cleandoc("""
    Your output MUST be a JSON object that conforms to the JSON Schema below. All
    JSON object property names MUST be enclosed in double quotes.

    You MUST take the types of the OUTPUT SCHEMA into account and adjust your
    provided text to fit the required types.

    Here is the OUTPUT SCHEMA:
    {{ response_model_json }}
""")

DEFAULT_PROMPT = inspect.cleandoc("""
    You are an expert at extracting entities from user provided text, data or
    information and always maintain as much semantic meaning as possible.
    Make sure to interpret numbers written as text as numbers when required. Make
    sure to identify separate entities.

    Analyze the provided input from the user and generate any entities or objects
    that match the requested JSON output according to the OUTPUT SCHEMA provided.

    Your output MUST be a JSON object that conforms to the JSON Schema below. All
    JSON object property names MUST be enclosed in double quotes.

    You MUST take the types of the OUTPUT SCHEMA into account and adjust your
    provided text to fit the required types.

    Here is the OUTPUT SCHEMA:
    {{ response_model_json }}

    {% if examples is defined %}
    Here are some examples to show what the desired output is:
    {{ examples }}
    {% endif -%}

    Input from user:
""")


class Prompt:
    """
    Class used to render a prompt template.
    """

    def __init__(self, prompt: str | None = None) -> None:
        """
        :param prompt: The prompt to use for the Forge. This can be jinja2 templated
                       text with placeholders which will be replaced with their values
                       when the 'render()' method is called.
        """
        self.prompt = prompt or DEFAULT_PROMPT  # Load default prompt if none provided

        # Create a jinja2 environment for the class instance.
        self._env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(default_for_string=False),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        self._prompt_variables: set[str] | None = None

    @property
    def prompt_variables(self) -> set[str]:
        """
        Get the variables used in the prompt. This property is cached.

        :return: The variables used in the prompt.
        """
        if self._prompt_variables:
            return self._prompt_variables

        if not self.prompt:
            return set()

        self._prompt_variables = meta.find_undeclared_variables(
            self._env.parse(self.prompt)
        )

        return self._prompt_variables

    def render(self, **kwargs: Any) -> str:
        """
        Render the prompt with the provided keyword arguments as context.

        :param kwargs: The keyword arguments to use to render the prompt.
        :return: The rendered prompt.
        """
        prompt_kwargs = self._process_kwargs(kwargs)

        # Build the prompt that should be rendered from the original prompt provided
        # as well as the user_input and response_model if it is required.
        prompt_to_render = self.prompt

        # If a response model is required but it is not currently mentioned in the
        # prompt then add a section to the prompt for it.
        if (
            "response_model_json" in prompt_kwargs
            and "response_model_json" not in self.prompt_variables
        ):
            prompt_to_render += f"\n{RESPONSE_MODEL_TEXT}\n"

        # If the user_input is not specified in the prompt already then simply add it to
        # the end of the prompt.
        if "user_input" in prompt_kwargs and "user_input" not in self.prompt_variables:
            prompt_to_render += f"\n{prompt_kwargs['user_input']}\n"

        # Render the prompt and log it for debugging purposes.
        rendered_prompt = self._env.from_string(prompt_to_render).render(
            **prompt_kwargs
        )
        logger.debug(f"Rendered prompt:\n{rendered_prompt}")

        return rendered_prompt

    def _process_kwargs(self, prompt_values: dict[str, Any]) -> dict[str, Any]:
        """
        Process the prompt kwargs to handle cases around response_model.

        :param prompt_values: The keyword arguments to use to render the prompt.
        :return: Processed keyword arguments
        """
        response_model: BaseModel | None = prompt_values.get("response_model")
        if response_model:
            # Add response model schema
            prompt_values["response_model_json"] = json.dumps(
                response_model.model_json_schema()
            )

        return prompt_values
