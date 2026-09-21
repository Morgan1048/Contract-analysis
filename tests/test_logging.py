import json

import pytest

from core.obs.logging import JsonFormatter
from core.obs.trace import span


def test_span_error_is_json_and_redacts_secret():
    import logging
    record = logging.LogRecord("x", logging.INFO, "", 1, "event", (), None)
    record.fields = {"message": "api_key=secret-value", "optional": None}
    data = json.loads(JsonFormatter().format(record))
    assert data["message"] == "[REDACTED]"
    assert "optional" not in data

    with pytest.raises(RuntimeError):
        with span("failing"):
            raise RuntimeError("boom")
