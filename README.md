<p align="center">
  <img src="modelsmith.png" style="width: auto; height: auto;"/>
</p>

# Modelsmith
### Modelsmith is a Python library that allows you to get structured responses in the form of Pydantic models and Python types from Anthropic, Google Gemini, and OpenAI models.

Currently it allows you to use the following classes of model:
- __AnthropicModel__ (used with Anthropic's full set of models)
- __OpenAIModel__ (used with OpenAI's full set of models)
- __GeminiModel__ (used with Google's full set of Gemini models)

Modelsmith provides a unified interface over all of these. It has been designed to be extensible and can adapt to other models in the future.

# Notable Features

- __Structured Responses__: Specify both Pydantic models and Python types as the outputs of your LLM responses.
- __Templating__: Use Jinja2 templating in your prompts to allow complex prompt logic.
- __Default and Custom Prompts__: A default prompt template is provided but you can also specify your own.
- __Retry Logic__: Number of retries is user configurable.
- __Validation__: Outputs from the LLM are validated against your requested response model. Errors are fed back to the LLM to try and correct any validation failures.

# Installation

Install Modelsmith using pip or your favourite python package manager.

`pip` example:
```bash
pip install modelsmith
```
# Documentation

For detailed documentation please have a look at [https://christo-olivier.github.io/modelsmith](https://christo-olivier.github.io/modelsmith)

# Get in touch

If you have any questions or suggestions, feel free to open an issue or start a discussion.

# License

This project is licensed under the terms of the MIT License.