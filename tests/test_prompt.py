import inspect

from modelsmith.prompt import Prompt


def test_render_prompt_user_input() -> None:
    """
    Test that the prompt is rendered correctly when user_input is passed. Thin this
    test {{ user_input }} is not in the prompt and so it should be added as a new line
    at the end of the prompt.
    """
    prompt = Prompt(prompt="Hello")
    assert prompt.render(user_input="World!") == "Hello\nWorld!"


def test_render_prompt_response_model_not_in_prompt() -> None:
    """
    Test that the prompt includes a section for the response model when no such section
    exists in the original prompt.
    """
    prompt_no_response_model = inspect.cleandoc("""
    This is a test.
    """)

    prompt = Prompt(prompt=prompt_no_response_model)
    result = prompt.render(response_model_json='{"test": 1}')
    assert result == inspect.cleandoc("""
    This is a test.
    Your output MUST be a JSON object that conforms to the JSON Schema below. All
    JSON object property names MUST be enclosed in double quotes.

    You MUST take the types of the OUTPUT SCHEMA into account and adjust your
    provided text to fit the required types.

    Here is the OUTPUT SCHEMA:
    {"test": 1}
    """)


def test_render_prompt_no_prompt_use_default() -> None:
    """
    Test that the the default prompt is used when no prompt is provided.
    """
    prompt = Prompt()
    result = prompt.render(response_model_json='{"test": 1}')
    assert result == inspect.cleandoc("""
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
        {"test": 1}

        Input from user:""")
