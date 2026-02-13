"""Tests for the FunASR client."""

import json
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from unittest.mock import patch, MagicMock

import pytest

from funasr_server.client import FunASR, ServerError


class MockRPCHandler(BaseHTTPRequestHandler):
    """Mock JSON-RPC server for testing."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        method = body.get("method")
        req_id = body.get("id")
        params = body.get("params", {})

        if method == "health":
            result = {"status": "ok", "loaded_models": [], "device": "cpu", "cuda_available": False}
        elif method == "load_model":
            result = {"name": params.get("name", "default"), "status": "loaded"}
        elif method == "unload_model":
            result = {"name": params.get("name", "default"), "status": "unloaded"}
        elif method == "infer":
            result = {"results": [{"key": "test", "text": "hello world"}]}
        elif method == "transcribe":
            result = {"results": [{"key": "test", "text": "hello world"}]}
        elif method == "list_models":
            result = {"models": {}}
        elif method == "execute":
            result = {"output": "ok", "return_value": None}
        elif method == "error_test":
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": "test error"}}
            self._send_json(resp)
            return
        else:
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
            self._send_json(resp)
            return

        resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
        self._send_json(resp)

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress output


@pytest.fixture
def mock_server():
    """Start a mock RPC server for testing."""
    server = HTTPServer(("127.0.0.1", 0), MockRPCHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture
def client(mock_server):
    """Create a FunASR client connected to the mock server."""
    c = FunASR(port=mock_server, host="127.0.0.1")
    c._process = None  # skip process management
    return c


def test_health(client):
    result = client.health()
    assert result["status"] == "ok"
    assert "loaded_models" in result


def test_load_model(client):
    result = client.load_model(model="test-model")
    assert result["status"] == "loaded"
    assert result["name"] == "default"


def test_load_model_with_name(client):
    result = client.load_model(model="test-model", name="my_model")
    assert result["name"] == "my_model"


def test_unload_model(client):
    result = client.unload_model()
    assert result["status"] == "unloaded"


def test_infer(client):
    result = client.infer(input="test.wav")
    assert len(result) == 1
    assert result[0]["text"] == "hello world"


def test_infer_with_bytes(client):
    result = client.infer(input_bytes=b"fake audio data")
    assert len(result) == 1


def test_infer_no_input(client):
    with pytest.raises(ValueError, match="Either 'input' or 'input_bytes'"):
        client.infer()


def test_transcribe(client):
    result = client.transcribe(audio="test.wav")
    assert len(result) == 1
    assert result[0]["text"] == "hello world"


def test_transcribe_with_bytes(client):
    result = client.transcribe(audio_bytes=b"fake audio data")
    assert len(result) == 1


def test_transcribe_no_input(client):
    with pytest.raises(ValueError, match="Either 'input' or 'input_bytes'"):
        client.transcribe()


def test_list_models(client):
    result = client.list_models()
    assert "models" in result


def test_execute(client):
    result = client.execute("x = 1")
    assert result["output"] == "ok"


def test_server_error(client):
    with pytest.raises(ServerError, match="test error"):
        client._rpc_call("error_test", {})


def test_connection_error():
    c = FunASR(port=1, host="127.0.0.1")
    with pytest.raises(ConnectionError):
        c._rpc_call("health", {}, timeout=1)


def test_context_manager_init():
    c = FunASR(runtime_dir="/tmp/test_funasr")
    assert c.runtime_dir == Path("/tmp/test_funasr")
    assert c.port == 0
    assert c.host == "127.0.0.1"


def test_is_running_no_process():
    c = FunASR()
    assert c.is_running() is False


def test_stop_no_process():
    c = FunASR()
    c.stop()  # should not raise
