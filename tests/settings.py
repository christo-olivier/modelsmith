import pytest

MODEL_SETTINGS_TEMP_ZERO = {"temperature": 0.0}
MODEL_SETTINGS_TEMP_POINT_NINE = {"temperature": 0.0}

MODEL_SETTINGS_PARAMS = [
    pytest.param(MODEL_SETTINGS_TEMP_ZERO, id="temp_zero"),
    pytest.param(MODEL_SETTINGS_TEMP_POINT_NINE, id="temp_point_nine"),
]
