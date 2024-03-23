<p align="center">
  <img src="modelsmith.png" style="width: auto; height: auto;"/>
</p>

# Modelsmith
### Modelsmith is a Python library that allows you to get structured responses in the form of Pydantic models and Python types from Google's Vertex AI models.

Currently it allows you to use three classes of model:
- __ChatModel__ (most commonly used with `chat-bison`)
- __TextGenerationModel__ (most commonly used with `text-bison`)
- __GenerativeModel__ (most commonly used with `gemini-pro`)

Modelsmith allows a unified interface over all of these. It has been designed to be extensible and can adapt to other models in the future.

# Notable Features

- __Structured Responses__: Specify both Pydantic models and Python types as the outputs of your LLM responses.
- __Templating__: Use Jinja2 templating in your prompts to allowing complex prompt logic.
- __Default and Custom Prompts__: A default prompt template is provided but you can also specify your own.
- __Retry Logic__: Number of retries is user configurable.
- __Validation__: Outputs from the LLM are validated against your requested response model. Errors are fed back to the LLM to try and correct any validation failures.

# Installation

Install Modelsmith using pip or your favourite python package manager.

`pip` example:
```bash
pip install modelsmith
```

## Google Cloud Authentication

Authentication to Google Cloud is done via the Application Default Credentials flow. So make sure you have ADC configured. See [Google's documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc) for more details.


# Getting started

## Extracting a Pydantic models

Lets look at an example of extracting a Pydantic model from some text.

```python
from modelsmith import Forge
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerativeModel


# Define the pydantic model you want to receive as the response
class User(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")
    country: str = Field(description="The country where the person lives")


# Create your forge instance
forge = Forge(model=GenerativeModel("gemini-1.0-pro"), response_model=User)

# Generate a User instance from the prompt
user = forge.generate("Terry Tate 60. Lives in Irvine, United States.")

print(user)  # name='Terry Tate' age=60 city='Irvine' country='United States'
```

## Extracting a combined Pydantic and Python type

Modelsmith does not restrict you to either Pydantic models or Python types. You can combine them in the same response. Below we extract a list of Pydantic model instances.

```python
from modelsmith import Forge
from pydantic import BaseModel, Field
from vertexai.generative_models import GenerativeModel


class City(BaseModel):
    city: str = Field(description="The name of the city")
    state: str = Field(description="2-letter abbreviation of the state")


# Pass a list of Pydantic models to the response_model argument.
forge = Forge(
    model=GenerativeModel("gemini-1.0-pro"),
    response_model=list[City],
)

response = forge.generate("I have lived in Irvine, CA and Dallas TX")

print(response)  # [City(city='Irvine', state='CA'), City(city='Dallas', state='TX')]
```

## Using different model types

Using a different Vertex AI model is as simple as passing it to the Forge. Taking the example above lets use `text-bison` instead of `gemini-pro`.

```python
from modelsmith import Forge
from pydantic import BaseModel, Field
from vertexai.language_models import TextGenerationModel  # import the correct class


class City(BaseModel):
    city: str = Field(description="The name of the city")
    state: str = Field(description="2-letter abbreviation of the state")


# text-bison instead of gemini-pro
forge = Forge(
    model=TextGenerationModel.from_pretrained("text-bison"),
    response_model=list[City],
)

response = forge.generate("I have lived in Irvine, CA and Dallas TX")

print(response)  # [City(city='Irvine', state='CA'), City(city='Dallas', state='TX')]
```

## Using the default prompt template

The previous examples use the built in prompt template in zero-shot mode. The default template also works in few-shot mode and allows you to pass in examples via the `prompt_values` parameter of the `generate` method. The default prompt template has a template variable called `examples` that we pass our example text to. The following example shows how this can be used.

```python
import inspect

from modelsmith import Forge
from vertexai.generative_models import GenerativeModel

# Create your forge instance
forge = Forge(model=GenerativeModel("gemini-1.0-pro"), response_model=list[str])

# Define examples, using inspect.cleandoc to remove indentation
examples = inspect.cleandoc("""
    input: John Doe is forty years old. Lives in Alton, England
    output: ["John Doe", "40", "Alton", "England"]

    input: Sarah Green lives in London, UK. She is 32 years old.
    output: ["Sarah Green", "32", "London", "UK"]
""")

# Generate a Python list of string values from the input text
response = forge.generate(
    "Sophia Schmidt twenty three. Resident in Berlin Germany.",
    prompt_values={"examples": examples},
)

print(response)  # ['Sophia Schmidt', '23', 'Berlin', 'Germany']
```

## Using your own prompt template

If you want to use your own prompt you can simply pass it to the `prompt` parameter of the `Forge` class. Any jinja2 template variables will be replaced with the values provided in the `prompt_values` parameter of the `generate` method.

⚠️ If using your own prompt include a jinja template variable called `response_model_json` to place your response model json schema in your preferred location. If `response_model_json` is not provided then the default response model template text will be appended to the end of your prompt.

Here is an example of using a custom prompt that includes the `response_model_json` template variable.

```python
import inspect

from modelsmith import Forge
from vertexai.language_models import TextGenerationModel

# Create your custom prompt
my_prompt = inspect.cleandoc("""
    You are extracting city names from user provided text. You are only to extract
    city names and you should ignore country names or any other entities that are not
    cities.

    You MUST take the types of the OUTPUT SCHEMA into account and adjust your
    provided text to fit the required types.

    Here is the OUTPUT SCHEMA:
    {{ response_model_json }}
""")

# Create your forge instance, passing your prompt
forge = Forge(
    model=TextGenerationModel.from_pretrained("text-bison"),
    response_model=list,
    prompt=my_prompt,
)

# Generate a your response
response = forge.generate(
    "Berlin is the capital of Germany. London is the capital of England."
)

print(response)  # ['Berlin', 'London']
```

The same example above would also work if the `response_model_json` was left out of the prompt due to this being added automatically if missing.

```python
import inspect

from modelsmith import Forge
from vertexai.language_models import TextGenerationModel

# Create your custom prompt
my_prompt = inspect.cleandoc("""
    You are extracting city names from user provided text. You are only to extract
    city names and you should ignore country names or any other entities that are not
    cities.
""")

# Create your forge instance, passing your prompt
forge = Forge(
    model=TextGenerationModel.from_pretrained("text-bison"),
    response_model=list,
    prompt=my_prompt,
)

# Generate a your response
response = forge.generate(
    "Berlin is the capital of Germany. London is the capital of England."
)

print(response)  # ['Berlin', 'London']
```

## Placing user_input inside your prompt

By default user input is appended to the end of both custom and default prompts. Modelsmith allows you to place user input anywhere inside your custom prompt by adding the template variable `{{ user_input }}` where you want the user input to go.

```python
# Create your custom prompt with user input placed at the beginning
my_prompt = inspect.cleandoc("""
    Consider the following user input: {{ user_input }}

    You are extracting numbers from user input and combing them into one number. 
    Take into account numbers written as text as well as in numerical format.
""")
```

## Setting the number of retries

By default Modelsmith will try to get the desired response model from the LLM three times before raising an exception. On each retry the validation error is fed back to the LLM with a request to correct it. 

You can change this by passing the `max_retries` parameter to the `Forge` class.

```python
# Create your forge instance, setting the number of retries
forge = Forge(
    model=GenerativeModel("gemini-1.0-pro"), response_model=int, max_retries=2
)
```

## Matching patterns

Modelsmith looks for JSON output in the LLM response. It uses regular expressions to identify JSON output. If for any reason you want to use a different pattern you can pass it to the `match_pattern` parameter of the `Forge` class.

## Failing silently

Modelsmith will raise a `ModelNotDerivedError` exception if no valid response was obtained. You can change this by passing `False` to the `raise_on_failure` parameter of the `Forge` class.

This will suppress the exception and return `None` instead.

## Passing prompt template variables and model settings

You can pass prompt template variables and model settings by passing them to the `prompt_values` and `model_settings` parameters of the `generate` method.


```python
import inspect

from modelsmith import Forge
from vertexai.generative_models import GenerativeModel

# Create your custom prompt
my_prompt = inspect.cleandoc("""
    You are extracting city names from user provided text. You are only to extract
    city names and you should ignore country names or any other entities that are not
    cities.

    {{ user_input_prefix }}
    {{ user_input }}
""")

# Create your forge instance, passing your prompt
forge = Forge(
    model=GenerativeModel("gemini-1.0-pro"),
    response_model=list,
    prompt=my_prompt,
    max_retries=2,
)

# Custom LLM settings
model_settings = {
    "temperature": 0.8,
    "top_p": 1.0,
}

# Prompt template variable values to pass
prompt_values = {
    "user_input_prefix": "I have a the following text to analyze: ",
}

# Generate a your response
response = forge.generate(
    "Berlin is the capital of Germany. London is the capital of England.",
    prompt_values=prompt_values,
    model_settings=model_settings,
)

print(response)  # ['Berlin', 'London']
```

## Learn more

Have a look at the tests included in this repository for more examples.

# Get in touch

If you have any questions or suggestions, feel free to open an issue or start a discussion.

# License

This project is licensed under the terms of the MIT License.