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

Authentication to Google Cloud is done via either:
- Application Default Credentials flow. See [Google's documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart?usertype=adc) for more details.
- Gemini API Key. See [Google Gemini API docs](https://ai.google.dev/gemini-api/docs/api-key?_gl=1*1ya8hy4*_up*MQ..*_ga*MTA2MDc3MjY2MC4xNzU3NzkwNDQz*_ga_P1DBVKWT6V*czE3NTc3OTgwNTMkbzMkZzAkdDE3NTc3OTgwNTMkajYwJGwwJGgxMDQzMjc4MTU2).

The `GeminiModel` allows you to pass the `vertexai`, `api_key`, `project`, and `location` when you initialize the class instance. If you do not pass this in it will be inferred from the environment variables `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` as per the documentation.

## Open AI Authentication
Authentication to OpenAI is done via the OpenAI flow. See the [OpenAI documentation](https://platform.openai.com/docs/quickstart/step-2-set-up-your-api-key) for more details.

The `OpenAIModel` allows you to pass the `api_key`, `organization` and `project` when you initialize the class instance. If you do not pass this in it will be inferred from the environment variables `OPENAI_API_KEY`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` as per the OpenAI documentation.