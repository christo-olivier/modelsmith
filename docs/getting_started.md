# Getting started

## Extracting a Pydantic model

Lets look at an example of extracting a Pydantic model from some text.

```python
from modelsmith import Forge, OpenAIModel
from pydantic import BaseModel, Field


# Define the pydantic model you want to receive as the response
class User(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")
    country: str = Field(description="The country where the person lives")


# Create your forge instance
forge = Forge(model=OpenAIModel("gpt-3.5-turbo"), response_model=User)

# Generate a User instance from the prompt
user = forge.generate("Terry Tate 60. Lives in Irvine, United States.")

print(user)  # name='Terry Tate' age=60 city='Irvine' country='United States'
```

## Extracting a combined Pydantic and Python type

Modelsmith does not restrict you to either Pydantic models or Python types. You can combine them in the same response. Below we extract a list of Pydantic model instances.

```python
from modelsmith import Forge, VertexAIGenerativeModel
from pydantic import BaseModel, Field


class City(BaseModel):
    city: str = Field(description="The name of the city")
    state: str = Field(description="2-letter abbreviation of the state")


# Pass a list of Pydantic models to the response_model argument.
forge = Forge(
    model=VertexAIGenerativeModel("gemini-1.5-pro"),
    response_model=list[City],
)

response = forge.generate("I have lived in Irvine, CA and Dallas TX")

print(response)  # [City(city='Irvine', state='CA'), City(city='Dallas', state='TX')]
```

## Using different model types

Using a different model is as simple as passing the desired model class to the Forge. Taking the example above lets use `text-bison` instead of `gemini-pro`.

```python
from modelsmith import Forge, VertexAITextGenerationModel  # import the correct class
from pydantic import BaseModel, Field


class City(BaseModel):
    city: str = Field(description="The name of the city")
    state: str = Field(description="2-letter abbreviation of the state")


# text-bison instead of gemini-pro
forge = Forge(
    model=VertexAITextGenerationModel("text-bison"),
    response_model=list[City],
)

response = forge.generate("I have lived in Irvine, CA and Dallas TX")

print(response)  # [City(city='Irvine', state='CA'), City(city='Dallas', state='TX')]
```

If we want to use an Anthropic model the same applies. Simply select the appropriate model class, specify which Anthropic model to use (in this case `claude-3-haiku-20240307`), and pass it to the `Forge` instance.

```python
from modelsmith import Forge, AnthropicModel  # import the correct class
from pydantic import BaseModel, Field


class City(BaseModel):
    city: str = Field(description="The name of the city")
    state: str = Field(description="2-letter abbreviation of the state")


# Anthropic's claude-3-haiku-20240307 instead of gemini-pro
forge = Forge(
    model=AnthropicModel("claude-3-haiku-20240307"),
    response_model=list[City],
)

response = forge.generate("I have lived in Irvine, CA and Dallas TX")

print(response)  # [City(city='Irvine', state='CA'), City(city='Dallas', state='TX')]
```

## Using the default prompt template

The previous examples use the built in prompt template in zero-shot mode. The default template also works in few-shot mode and allows you to pass in examples via the `prompt_values` parameter of the `generate` method. The default prompt template has a template variable called `examples` that we pass our example text to. The following example shows how this can be used.

```python
import inspect

from modelsmith import Forge, VertexAIGenerativeModel

# Create your forge instance
forge = Forge(
    model=VertexAIGenerativeModel("gemini-1.5-flash"), response_model=list[str]
)

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

from modelsmith import Forge, OpenAIModel

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
    model=OpenAIModel("gpt-4o"),
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

from modelsmith import Forge, VertexAITextGenerationModel

# Create your custom prompt
my_prompt = inspect.cleandoc("""
    You are extracting city names from user provided text. You are only to extract
    city names and you should ignore country names or any other entities that are not
    cities.
""")

# Create your forge instance, passing your prompt
forge = Forge(
    model=VertexAITextGenerationModel("text-bison"),
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
    model=VertexAIGenerativeModel("gemini-1.0-pro"), response_model=int, max_retries=2
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

from modelsmith import Forge, OpenAIModel

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
    model=OpenAIModel("gpt-4o"),
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