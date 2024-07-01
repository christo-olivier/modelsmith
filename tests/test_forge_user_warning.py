import inspect
from typing import Any

import pytest
from modelsmith import Forge
from modelsmith.language_models import (
    OpenAIModel,
    VertexAIChatModel,
    VertexAIGenerativeModel,
    VertexAITextGenerationModel,
)
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import ChatModel, TextGenerationModel

from tests.models import User
from tests.settings import MODEL_INSTANCE_PARAMS, MODEL_SETTINGS_PARAMS


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_few_shot_pydantic_model(
    model: ChatModel
    | GenerativeModel
    | TextGenerationModel
    | VertexAIChatModel
    | VertexAIGenerativeModel
    | VertexAITextGenerationModel
    | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    This test ensures the Forge raises the appropriate deprecation warning when
    using the old style models directly from the Vertex AI SDK.
    """

    examples = inspect.cleandoc("""
    input: John Doe is forty years old. Lives in Alton, England
    output: '{"name":"John Doe", "age":40, "city": "Alton", "country": "England"}'

    input: Sarah Green lives in London, UK. She is 32 years old.
    output: '{"name":"Sarah Green","age":32,"city":"London","country":"UK"}'
    """)
    if isinstance(model, (ChatModel, GenerativeModel, TextGenerationModel)):
        with pytest.warns(
            DeprecationWarning,
            match=(
                "Using VertexAI classes directly is deprecated and will be removed in "
                "a future release. Please import the classes from the `language_model` "
                "module instead."
            ),
        ):
            forge = Forge(model=model, response_model=User)
    else:
        forge = Forge(model=model, response_model=User)

    response = forge.generate(
        "Terry Tate 60. Lives in Irvine, United States.",
        prompt_values={"examples": examples},
        model_settings=model_settings,
    )

    expected = User(name="Terry Tate", age=60, city="Irvine", country="United States")

    assert response == expected
