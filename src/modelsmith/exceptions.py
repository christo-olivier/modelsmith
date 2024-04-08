from typing import Sequence


class CombinedException(Exception):
    """
    Exception used to aggregate multiple exceptions and raise them once.
    """

    def __init__(self, exceptions: Sequence[Exception]):
        self.exceptions = exceptions

    def __str__(self) -> str:
        return ", ".join(str(e) for e in self.exceptions)


class ModelNotDerivedError(Exception):
    """
    Exception raised when a pydantic model could not be created from the response of
    the LLM.
    """

    pass


class PatternNotFound(Exception):
    """
    Exception used when a pattern being searched for has not been found.
    """

    pass
