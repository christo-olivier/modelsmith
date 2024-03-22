import types
from functools import cached_property
from typing import Generic, TypeVar

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from modelsmith.enums import ResponseModelType

T = TypeVar("T")


class ResponseModel(Generic[T]):
    """
    Class used to create the response model that used by the Forge. It provides
    functionality to deal with both Pydantic and plain Python response models.
    """

    def __init__(self, response_model: type[T]) -> None:
        self.model = response_model
        self.pydantic_model = self._process_model(response_model)

    @cached_property
    def original_type(
        self,
    ) -> ResponseModelType:
        """
        Provides the type of response model, either Pydantic or Python.
        """
        # Note: checking that `self.model` is not of types.GenericAlias is done
        # because these types are not supported by `issubclass` in Python 3.10
        if (
            not isinstance(self.model, types.GenericAlias)
            and isinstance(self.model, type)
            and issubclass(self.model, BaseModel)
        ):
            return ResponseModelType.PYDANTIC

        return ResponseModelType.PYTHON

    def _process_model(self, response_model: type[T]) -> type[BaseModel]:
        """
        If the response_model is not already a Pydantic model, create one from the
        Python type that was passed in.

        :param response_model: The response model type.
        :return: A pydantic model class.
        """
        # If a pydantic class was passed then simply return it
        # Note: checking that `response_model` is not of types.GenericAlias is done
        # because these types are not supported by `issubclass` in Python 3.10
        if (
            not isinstance(response_model, types.GenericAlias)
            and isinstance(response_model, type)
            and issubclass(response_model, BaseModel)
        ):
            return response_model

        # If a python type was passed then create a pydantic model with a field called
        # value with that type
        metadata = getattr(response_model, "__metadata__", [])
        if isinstance(next(iter(metadata), None), FieldInfo):
            field_info = next(iter(metadata))
        else:
            field_info = FieldInfo(
                description="JSON schema the response should adhere to."
            )

        pydantic_model = create_model(
            "ResponseModel", value=(response_model, field_info)
        )

        return pydantic_model
