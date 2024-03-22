from enum import Enum


# Use subclassing of str and Enum instead of StrEnum to allow Python 3.10
# support
class ResponseModelType(str, Enum):
    PYDANTIC = "pydantic"
    PYTHON = "python"

    def __str__(self) -> str:
        return str.__str__(self)
