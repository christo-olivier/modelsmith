## Welcome to Modelsmith

Modelsmith is a Python library that allows you to get structured responses in the form of Pydantic models and Python types from Google Vertex AI and OpenAI models.

Currently it allows you to use the following classes of model:

- __AnthropicModel__ (used with Anthropic's full set of models)

- __OpenAIModel__ (used with OpenAI's full set of models)

- __GeminiModel__ (used with Google's full set of Gemini models)


Modelsmith provides a unified interface over all of these. It has been designed to be extensible and can adapt to other models in the future.

## Notable Features

:octicons-sparkle-fill-24: __Structured Responses__: Specify both Pydantic models and Python types as the outputs of your LLM responses.

:octicons-sparkle-fill-24: __Templating__: Use Jinja2 templating in your prompts to allow complex prompt logic.

:octicons-sparkle-fill-24: __Default and Custom Prompts__: A default prompt template is provided but you can also specify your own.

:octicons-sparkle-fill-24: __Retry Logic__: Number of retries is user configurable.

:octicons-sparkle-fill-24: __Validation__: Outputs from the LLM are validated against your requested response model. Errors are fed back to the LLM to try and correct any validation failures.
