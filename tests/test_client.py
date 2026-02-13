"""Tests for the FunASR client."""

import json
import signal
import subprocess
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
            result = {"name": params.get("name", "default"), "status": "loaded",
                       "hub": params.get("hub"), "device": params.get("device"),
                       "vad_model": params.get("vad_model")}
            result = {k: v for k, v in result.items() if v is not None}
            result["status"] = "loaded"
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
        elif method == "download_model":
            result = {"model": params.get("model"), "path": "/tmp/model", "hub": params.get("hub", "ms")}
        elif method == "error_test":
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": "test error"}}
            self._send_json(resp)
            return
        elif method == "error_with_data":
            resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": "test error", "data": "traceback info"}}
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


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

def test_init_defaults():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_funasr"
        c = FunASR(runtime_dir=str(test_dir))
        assert c.runtime_dir == test_dir.resolve()
        assert c.port == 0
        assert c.host == "127.0.0.1"
        assert c._process is None


def test_is_running_no_process():
    c = FunASR()
    assert c.is_running() is False


def test_is_running_with_live_process():
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    c._process = mock_proc
    assert c.is_running() is True


def test_is_running_with_dead_process():
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # exited
    c._process = mock_proc
    assert c.is_running() is False


def test_stop_no_process():
    c = FunASR()
    c.stop()  # should not raise


def test_stop_graceful():
    """stop() sends shutdown RPC then waits for process exit."""
    c = FunASR()
    mock_proc = MagicMock()
    c._process = mock_proc

    with patch.object(c, "_rpc_call"):
        c.stop()

    mock_proc.wait.assert_called_once()
    assert c._process is None


def test_stop_force_kill_windows():
    """stop() calls terminate() on Windows when process doesn't exit."""
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
    c._process = mock_proc

    with patch.object(c, "_rpc_call"), \
         patch("platform.system", return_value="Windows"):
        c.stop()

    mock_proc.terminate.assert_called_once()
    assert c._process is None


@pytest.mark.skipif(
    not hasattr(signal, "SIGKILL"),
    reason="SIGKILL only available on Unix",
)
def test_stop_force_kill_unix():
    """stop() sends SIGKILL on Unix when process doesn't exit."""
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
    c._process = mock_proc

    with patch.object(c, "_rpc_call"), \
         patch("platform.system", return_value="Linux"):
        c.stop()

    mock_proc.send_signal.assert_called_once_with(signal.SIGKILL)
    assert c._process is None


def test_ensure_installed_already_installed():
    c = FunASR()
    with patch.object(c.installer, "is_installed", return_value=True):
        result = c.ensure_installed()
        assert result is True


def test_ensure_installed_fresh():
    c = FunASR()
    with patch.object(c.installer, "is_installed", return_value=False), \
         patch.object(c.installer, "install"):
        result = c.ensure_installed()
        assert result is False


def test_start_no_uv():
    """start() raises if uv is not found."""
    c = FunASR()
    with patch.object(c.installer, "get_uv_path", return_value=None):
        with pytest.raises(RuntimeError, match="uv not found"):
            c.start()


def test_start_already_running():
    """start() returns early if server is already running."""
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # alive
    c._process = mock_proc
    c.port = 1234

    result = c.start()
    assert result == 1234


# ------------------------------------------------------------------
# RPC API via mock server
# ------------------------------------------------------------------

def test_health(client):
    result = client.health()
    assert result["status"] == "ok"
    assert "loaded_models" in result
    assert "cuda_available" in result


def test_load_model(client):
    result = client.load_model(model="test-model")
    assert result["status"] == "loaded"
    assert result["name"] == "default"


def test_load_model_with_name(client):
    result = client.load_model(model="test-model", name="my_model")
    assert result["name"] == "my_model"


def test_load_model_with_kwargs(client):
    """Extra kwargs (hub, device, etc.) are passed through."""
    result = client.load_model(model="test-model", hub="hf", device="cpu")
    assert result["status"] == "loaded"
    assert result["hub"] == "hf"
    assert result["device"] == "cpu"


def test_load_model_with_vad_model(client):
    """vad_model parameter is passed through."""
    result = client.load_model(model="test-model", vad_model="fsmn-vad")
    assert result["vad_model"] == "fsmn-vad"


def test_unload_model(client):
    result = client.unload_model()
    assert result["status"] == "unloaded"


def test_unload_model_by_name(client):
    result = client.unload_model(name="my_model")
    assert result["name"] == "my_model"
    assert result["status"] == "unloaded"


def test_infer(client):
    result = client.infer(input="test.wav")
    assert len(result) == 1
    assert result[0]["text"] == "hello world"


def test_infer_with_bytes(client):
    result = client.infer(input_bytes=b"fake audio data")
    assert len(result) == 1


def test_infer_with_name(client):
    """infer() passes model name."""
    result = client.infer(input="test.wav", name="vad")
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


def test_download_model(client):
    result = client.download_model(model="iic/SenseVoiceSmall")
    assert result["model"] == "iic/SenseVoiceSmall"
    assert result["path"] == "/tmp/model"


def test_download_model_with_hub(client):
    result = client.download_model(model="test-model", hub="hf")
    assert result["hub"] == "hf"


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_server_error(client):
    with pytest.raises(ServerError, match="test error"):
        client._rpc_call("error_test", {})


def test_server_error_with_data(client):
    """ServerError includes data field from server response."""
    with pytest.raises(ServerError) as exc_info:
        client._rpc_call("error_with_data", {})
    assert exc_info.value.code == -32000
    assert exc_info.value.data == "traceback info"


def test_server_error_attributes():
    """ServerError stores code and data."""
    err = ServerError(-32000, "test message", "extra data")
    assert err.code == -32000
    assert err.data == "extra data"
    assert str(err) == "test message"


def test_connection_error():
    c = FunASR(port=1, host="127.0.0.1")
    with pytest.raises(ConnectionError):
        c._rpc_call("health", {}, timeout=1)


def test_rpc_id_increments(client):
    """Each RPC call increments the request ID."""
    initial = client._rpc_id
    client.health()
    assert client._rpc_id == initial + 1
    client.health()
    assert client._rpc_id == initial + 2
