import io
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import app
import mcp_server
from core.tools import ToolRegistry


def test_stdio_continues_after_malformed_frame():
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    following = {**request, "id": 2}
    stdin = io.StringIO(json.dumps(request) + "\n{broken\n" + json.dumps(following) + "\n")
    stdout = io.StringIO()
    registry = ToolRegistry()

    mcp_server.serve_stdio(registry, stdin, stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses == [
        {"jsonrpc": "2.0", "id": 1, "result": {"tools": registry.mcp_specs()}},
        {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": registry.mcp_specs()}},
    ]


@pytest.mark.parametrize("message, expected_id", [
    ({"id": 0, "method": "tools/list"}, 0),
    ({"id": "request-1", "method": "tools/list"}, "request-1"),
    ({"id": None, "method": "tools/list"}, None),
    ({"method": "tools/list"}, None),
    ([], None),
])
def test_stdio_continues_after_dispatch_exception(message, expected_id, monkeypatch):
    dispatch = mcp_server.dispatch
    calls = 0

    def failing_dispatch(request, registry):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("private failure details")
        return dispatch(request, registry)

    monkeypatch.setattr(mcp_server, "dispatch", failing_dispatch)
    following = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    stdin = io.StringIO(json.dumps(message) + "\n" + json.dumps(following) + "\n")
    stdout = io.StringIO()
    registry = ToolRegistry()

    mcp_server.serve_stdio(registry, stdin, stdout)

    assert [json.loads(line) for line in stdout.getvalue().splitlines()] == [
        {"jsonrpc": "2.0", "id": expected_id,
         "error": {"code": -32603, "message": "internal error"}},
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": registry.mcp_specs()}},
    ]


@pytest.mark.parametrize("failure_stage", [None, "upload", "parse", "analyze", "save"])
def test_uploaded_temp_file_is_deleted_after_callback(failure_stage, tmp_path, monkeypatch):
    payload = b"%PDF-synthetic-upload"
    failure = RuntimeError(f"{failure_stage} failed")
    paths = []
    stages = []
    named_temporary_file = app.tempfile.NamedTemporaryFile
    result = SimpleNamespace(run_id="upload-run", incomplete=False, usage={})
    index = object()
    state = {}

    def create_temp(**kwargs):
        handle = named_temporary_file(dir=tmp_path, **kwargs)
        paths.append(app.Path(handle.name))
        return handle

    def check_stage(stage):
        stages.append(stage)
        assert paths[0].exists()
        if stage != "upload":
            assert paths[0].read_bytes() == payload
        if failure_stage == stage:
            raise failure

    def getvalue():
        check_stage("upload")
        return payload

    def parse(path):
        assert path == paths[0]
        check_stage("parse")
        return SimpleNamespace(chunks=[])

    def analyze(path, _client, **kwargs):
        assert path == paths[0]
        assert kwargs == {"index": index, "allow_remote": False}
        check_stage("analyze")
        return result

    def save(saved):
        assert saved is result
        check_stage("save")

    monkeypatch.setattr(app.tempfile, "NamedTemporaryFile", create_temp)
    monkeypatch.setattr(app, "st", SimpleNamespace(
        session_state=state,
        status=lambda *args, **kwargs: nullcontext(SimpleNamespace(update=lambda **kwargs: None)),
    ))
    monkeypatch.setattr(app, "parse_pdf", parse)
    monkeypatch.setattr(app, "ChunkIndex", lambda *args, **kwargs: index)
    monkeypatch.setattr(app.ProviderConfig, "from_env", lambda *args: object())
    monkeypatch.setattr(app, "LLMClient", lambda *args: object())
    monkeypatch.setattr(app, "analyze", analyze)
    monkeypatch.setattr(app, "_store", lambda: SimpleNamespace(save_run=save))
    monkeypatch.setattr(app, "build_tool_registry", lambda *args, **kwargs: object())

    try:
        if failure_stage is None:
            app._analyze_uploaded_file(SimpleNamespace(getvalue=getvalue))
            assert stages == ["upload", "parse", "analyze", "save"]
            assert state["result"] is result
            assert state["index"] is index
        else:
            with pytest.raises(RuntimeError) as raised:
                app._analyze_uploaded_file(SimpleNamespace(getvalue=getvalue))
            assert raised.value is failure
            assert stages[-1] == failure_stage
        assert len(paths) == 1
        assert not paths[0].exists()
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
