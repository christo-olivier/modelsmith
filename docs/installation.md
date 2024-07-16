# Installation

Install Modelsmith using pip or your favourite python package manager.

`pip` example:
```bash
pip install modelsmith
```

## Anthropic Authentication

Authentication to Anthropic is done via the Anthropic flow. See the [Anthropic documentation](https://docs.anthropic.com/en/docs/quickstart#set-your-api-key) for more details. 

The `AnthropicModel` class takes an optional `api_key` parameter. If not provided, the `ANTHROPIC_API_KEY` environment variable will be used.

## Google Cloud Authentication

Authentication to Google Cloud is done via the Application Default Credentials flow. So make sure you have ADC configured. See [Google's documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) for more details.

## Open AI Authentication
Authentication to OpenAI is done via the OpenAI flow. See the [OpenAI documentation](https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key) for more details.

The `OpenAIModel` allows you to pass the `api_key`, `organization` and `project` when you initialize the class instance. If you do not pass this in it will be inferred from the environment variables `OPENAI_API_KEY`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` as per the OpenAI documentation.