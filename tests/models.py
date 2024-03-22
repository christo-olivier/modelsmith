from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")
    city: str = Field(description="The city where the person lives")
    country: str = Field(description="The country where the person lives")


class City(BaseModel):
    city: str = Field(description="The name of the city")
    state: str = Field(description="2-letter abbreviation of the state")
