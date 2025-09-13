import inspect
from typing import Any

import pytest

from modelsmith import Forge, ResponseNotDerivedError
from modelsmith.language_models import (
    AnthropicModel,
    GeminiModel,
    OpenAIModel,
)
from tests.models import City, User
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
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    This test ensures the Forge works as expected with a ChatModel when trying
    to extract an entity using a few-shot prompt approach.
    """

    examples = inspect.cleandoc("""
    input: John Doe is forty years old. Lives in Alton, England
    output: '{"name":"John Doe", "age":40, "city": "Alton", "country": "England"}'

    input: Sarah Green lives in London, UK. She is 32 years old.
    output: '{"name":"Sarah Green","age":32,"city":"London","country":"UK"}'
    """)
    forge = Forge(model=model, response_model=User)

    response = forge.generate(
        "Terry Tate 60. Lives in Irvine, United States.",
        prompt_values={"examples": examples},
        model_settings=model_settings,
    )

    expected = User(name="Terry Tate", age=60, city="Irvine", country="United States")

    assert response == expected


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_zero_shot_pydantic_model(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    This test ensures the Forge works as expected with a ChatModel when trying
    to extract an entity using a few-shot prompt approach.
    """
    forge = Forge(model=model, response_model=User)

    response = forge.generate(
        "Terry Tate 60. Lives in Irvine, United States.", model_settings=model_settings
    )

    expected = User(name="Terry Tate", age=60, city="Irvine", country="United States")

    assert response == expected


@pytest.mark.sequential
def test_forge_raise_on_failure_exception() -> None:
    """
    This test checks that when a model cannot be derived and the `raise_on_failure` flag
    is set to True, that a ModelNotDerivedError is raised.
    """
    with pytest.raises(expected_exception=ResponseNotDerivedError):
        forge = Forge(
            model=GeminiModel("text-bison"),
            response_model=User,
            max_retries=1,
        )

        forge.generate(
            "Terry Tate United States.",
            model_settings={"temperature": 0.0},
        )


@pytest.mark.sequential
def test_forge_raise_on_failure_no_exception() -> None:
    """
    This test checks that when a model cannot be derived and the `raise_on_failure` flag
    is set to False, None is return as the value instead of an exception being raised.
    """
    forge = Forge(
        model=GeminiModel("text-bison"),
        response_model=User,
        max_retries=1,
        raise_on_failure=False,
    )

    result = forge.generate(
        "Terry Tate United States.",
        model_settings={"temperature": 0.0},
    )

    assert result is None


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_few_shot_python_list(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python list of string values from the input text.
    """
    examples = inspect.cleandoc("""
        input: John Doe is forty years old. Lives in Alton, England
        output: ["John Doe", "40", "Alton", "England"]

        input: Sarah Green lives in London, UK. She is 32 years old.
        output: ["Sarah Green", "32", "London", "UK"]
    """)

    forge = Forge(
        model=model,
        response_model=list[str],
    )

    response = forge.generate(
        "Terry Tate 60. Lives in Irvine in the United States.",
        prompt_values={"examples": examples},
        model_settings=model_settings,
    )

    assert response == ["Terry Tate", "60", "Irvine", "United States"]


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_zero_shot_python_list(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python list of string values from the input text.
    """
    forge = Forge(
        model=model,
        response_model=list[str],
    )

    response = forge.generate(
        "Terry Tate, 60. Lives in Irvine in the United States.",
        model_settings=model_settings,
    )

    assert response == ["Terry Tate", "60", "Irvine", "United States"]


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_zero_shot_python_list_integers(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python list of integers from a text input
    containing numbers written as strings and integers.
    """
    forge = Forge(
        model=model,
        response_model=list[int],
    )

    response = forge.generate(
        "one, two, three, 4",
        model_settings=model_settings,
    )

    assert response == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_few_shot_python_list_floats(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python list of floats from a text input containing
    numbers written as strings and floats.
    """
    examples = inspect.cleandoc("""
        input: I took three tests and scored 1.5, 2.3, 3.7
        output: [3.0, 1.5, 2.3, 3.7]

        input: I ate 2 apples and drank one orange juice.
        output: [2.0, 1.0]
    """)

    forge = Forge(
        model=model,
        response_model=list[float],
    )

    response = forge.generate(
        "I took two books from the library and kept them for 2.5 and 3.5 months.",
        prompt_values={"examples": examples},
        model_settings=model_settings,
    )

    assert response == [2.0, 2.5, 3.5]


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_zero_shot_python_list_floats(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python list of floats from the numbers encountered
    in a text input.
    """
    forge = Forge(
        model=model,
        response_model=list[float],
    )

    response = forge.generate(
        "I bought a pasty for £10 and got £2.50 change",
        model_settings=model_settings,
    )

    assert response == [10.0, 2.50]


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_few_shot_python_float(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python float from the number written as a word.
    """
    examples = inspect.cleandoc("""
        Make sure to convert numbers written as strings to floats.

        input: I took three tests.
        output: 3.0

        input: I ate 2 apples.
        output: 2.0
    """)

    forge = Forge(
        model=model,
        response_model=float,
    )

    response = forge.generate(
        "I took two books from the library",
        prompt_values={"examples": examples},
        model_settings=model_settings,
    )

    assert response == 2.0


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_zero_shot_python_float(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python float from the number written as a word.
    """
    forge = Forge(
        model=model,
        response_model=float,
    )

    response = forge.generate(
        "I bought one pasty.",
        model_settings=model_settings,
    )

    assert response == 1.0


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_few_shot_python_integer(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python integer from the input text where a float is
    provided.
    """
    examples = inspect.cleandoc("""
        Make sure to convert numbers written as strings to integers and convert 
        float numbers to integers by rounding down to the nearest integer.

        input: I took three tests.
        output: 3

        input: I ate 2.5 apples.
        output: 2
    """)

    forge = Forge(
        model=model,
        response_model=int,
    )

    response = forge.generate(
        "I took 4.5 loafs of bread from the bakery",
        prompt_values={"examples": examples},
        model_settings=model_settings,
    )

    assert response == 4


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_zero_shot_python_integer(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python integer with rounding. At present it seems that
    the model is rounding up at half values. So 1.5 will be returned as 2 rather than 1.
    """
    forge = Forge(
        model=model,
        response_model=int,
    )

    response = forge.generate(
        "I bought 1.25 pasties.",
        model_settings=model_settings,
    )

    assert response == 1


@pytest.mark.parametrize(
    "model",
    MODEL_INSTANCE_PARAMS,
)
@pytest.mark.parametrize(
    "model_settings",
    MODEL_SETTINGS_PARAMS,
)
@pytest.mark.sequential
def test_forge_zero_shot_list_pydantic_location(
    model: AnthropicModel | GeminiModel | OpenAIModel,
    model_settings: dict[str, Any],
) -> None:
    """
    Test that the model returns a python list of pydantic City models from the input
    text.
    """
    forge = Forge(
        model=model,
        response_model=list[City],
    )

    response = forge.generate(
        "I have lived in Irvine, CA and Dallas, TX",
        model_settings=model_settings,
    )

    assert response == [
        City(city="Irvine", state="CA"),
        City(city="Dallas", state="TX"),
    ]
