## Welcome to Modelsmith

Modelsmith is a Python library that allows you to get structured responses in the form of Pydantic models and Python types from Google Vertex AI and OpenAI models.

Currently it allows you to use the following classes of model:

- __AnthropicModel__ (used with Anthropic's set of models such as `claude-3-haiku`, `claude-3-sonnet`, `claude-3-opus` and `claude-3_5-sonnet`)

- __OpenAIModel__ (used with OpenAI's set of models such as `gpt-3.5-turbo`, `gpt-4` and `gpt-4o`)

- __VertexAIChatModel__ (used with Google Vertex AI's chat models such as `chat-bison`)

- __VertexAITextGenerationModel__ (used with Google Vertex AI's text generation models such as `text-bison`)

- __VertexAIGenerativeModel__ (used with Google Vertex AI's generative models such as `gemini-pro`)


Modelsmith provides a unified interface over all of these. It has been designed to be extensible and can adapt to other models in the future.

## Notable Features

:octicons-sparkle-fill-24: __Structured Responses__: Specify both Pydantic models and Python types as the outputs of your LLM responses.

:octicons-sparkle-fill-24: __Templating__: Use Jinja2 templating in your prompts to allow complex prompt logic.

:octicons-sparkle-fill-24: __Default and Custom Prompts__: A default prompt template is provided but you can also specify your own.

:octicons-sparkle-fill-24: __Retry Logic__: Number of retries is user configurable.

:octicons-sparkle-fill-24: __Validation__: Outputs from the LLM are validated against your requested response model. Errors are fed back to the LLM to try and correct any validation failures.
