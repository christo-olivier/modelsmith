import time
from typing import Any, Generator

import pytest


@pytest.fixture(autouse=True)
def slow_down_tests() -> Generator[Any, Any, Any]:
    yield
    time.sleep(1)
