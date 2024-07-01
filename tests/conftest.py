import os
import threading
from typing import Any, Generator

import pytest
from dotenv import load_dotenv

# Create a lock object
sequential_lock = threading.Lock()


@pytest.fixture(autouse=True)
def run_sequentially(request: pytest.FixtureRequest) -> Generator[Any, Any, Any]:
    if request.node.get_closest_marker("sequential"):
        with sequential_lock:
            yield
    else:
        yield


@pytest.fixture(autouse=True)
def load_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # Load environment variables from the .env file
    load_dotenv()

    # Set the environment variables for the tests
    for key, value in os.environ.items():
        monkeypatch.setenv(key, value)
