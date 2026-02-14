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
            # Echo back all params so tests can verify what was sent
            result = dict(params)
            result["status"] = "loaded"
        elif method == "unload_model":
            result = {"name": params.get("name", "default"), "status": "unloaded"}
        elif method == "infer":
            # Echo back params + results
            result = {"results": [{"key": "test", "text": "hello world"}], "params_echo": params}
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
        elif method == "bad_jsonrpc":
            resp = {"jsonrpc": "1.0", "id": req_id, "result": {}}
            self._send_json(resp)
            return
        elif method == "bad_id":
            resp = {"jsonrpc": "2.0", "id": 999999, "result": {}}
            self._send_json(resp)
            return
        elif method == "no_result":
            resp = {"jsonrpc": "2.0", "id": req_id}
            self._send_json(resp)
            return
        elif method == "malformed_error":
            resp = {"jsonrpc": "2.0", "id": req_id, "error": "not a dict"}
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
# RPC API via mock server — load_model
# ------------------------------------------------------------------

def test_health(client):
    result = client.health()
    assert result["status"] == "ok"
    assert "loaded_models" in result
    assert "cuda_available" in result


def test_load_model(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        result = client.load_model(model="test-model")
    assert result["status"] == "loaded"
    assert result["name"] == "default"


def test_load_model_with_name(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        result = client.load_model(model="test-model", name="my_model")
    assert result["name"] == "my_model"


def test_load_model_auto_resolves_model_id(client):
    """load_model automatically resolves model name to hub-specific ID."""
    with patch("funasr_server.client.get_hub", return_value="hf"):
        result = client.load_model(model="SenseVoiceSmall")
    assert result["model"] == "FunAudioLLM/SenseVoiceSmall"
    assert result["hub"] == "hf"


def test_load_model_auto_resolves_ms(client):
    """load_model resolves to ModelScope ID when hub=ms."""
    with patch("funasr_server.client.get_hub", return_value="ms"):
        result = client.load_model(model="SenseVoiceSmall")
    assert result["model"] == "iic/SenseVoiceSmall"
    assert result["hub"] == "ms"


def test_load_model_explicit_hub(client):
    """Explicit hub parameter overrides auto-detection."""
    result = client.load_model(model="SenseVoiceSmall", hub="hf")
    assert result["model"] == "FunAudioLLM/SenseVoiceSmall"
    assert result["hub"] == "hf"


def test_load_model_resolves_vad_model(client):
    """vad_model is also resolved through the model registry."""
    result = client.load_model(model="SenseVoiceSmall", vad_model="fsmn-vad", hub="hf")
    assert result["vad_model"] == "fsmn-vad"


def test_load_model_resolves_sub_models(client):
    """punc_model and spk_model are also resolved."""
    result = client.load_model(
        model="SenseVoiceSmall",
        punc_model="ct-punc",
        spk_model="cam++",
        hub="ms",
    )
    assert result["punc_model"] == "ct-punc"
    assert result["spk_model"] == "cam++"


def test_load_model_with_device(client):
    result = client.load_model(model="test-model", device="cpu", hub="ms")
    assert result["device"] == "cpu"


def test_load_model_with_batch_size(client):
    """batch_size named param is sent to server."""
    result = client.load_model(model="test-model", batch_size=4, hub="ms")
    assert result["batch_size"] == 4


def test_load_model_with_fp16(client):
    """fp16 named param is sent to server."""
    result = client.load_model(model="test-model", fp16=True, hub="ms")
    assert result["fp16"] is True


def test_load_model_with_quantize(client):
    """quantize named param is sent to server."""
    result = client.load_model(model="test-model", quantize=True, hub="ms")
    assert result["quantize"] is True


def test_load_model_with_disable_update(client):
    """disable_update named param is sent to server."""
    result = client.load_model(model="test-model", disable_update=True, hub="ms")
    assert result["disable_update"] is True


def test_load_model_none_params_not_sent(client):
    """None params should NOT be included in params sent to server."""
    result = client.load_model(model="test-model", hub="ms")
    # batch_size, fp16, quantize etc. should not be in result since they were None
    assert "batch_size" not in result
    assert "fp16" not in result
    assert "quantize" not in result
    assert "disable_update" not in result
    assert "device" not in result
    assert "vad_model" not in result
    assert "punc_model" not in result
    assert "spk_model" not in result


def test_load_model_kwargs_passed_through(client):
    """Extra **kwargs are forwarded to the server."""
    result = client.load_model(
        model="test-model",
        hub="ms",
        trust_remote_code=True,
        ncpu=4,
    )
    assert result["trust_remote_code"] is True
    assert result["ncpu"] == 4


# ------------------------------------------------------------------
# RPC API via mock server — infer
# ------------------------------------------------------------------

def test_infer_audio(client):
    """infer(audio=...) sends file path as 'input'."""
    result = client.infer(audio="test.wav")
    assert len(result) == 1
    assert result[0]["text"] == "hello world"


def test_infer_text(client):
    """infer(text=...) sends text string as 'input'."""
    result = client.infer(text="你好世界")
    assert len(result) == 1
    assert result[0]["text"] == "hello world"


def test_infer_audio_bytes(client):
    """infer(audio_bytes=...) sends base64-encoded data."""
    result = client.infer(audio_bytes=b"fake audio data")
    assert len(result) == 1


def test_infer_with_name(client):
    """infer() passes model name."""
    result = client.infer(audio="test.wav", name="vad")
    assert len(result) == 1


def test_infer_no_input(client):
    """infer() raises when no input is provided."""
    with pytest.raises(ValueError, match="Provide exactly one of"):
        client.infer()


def test_infer_with_language(client):
    """infer(language=...) sends language param."""
    result = client.infer(audio="test.wav", language="zh")
    assert len(result) == 1


def test_infer_with_use_itn(client):
    """infer(use_itn=...) sends use_itn param."""
    result = client.infer(audio="test.wav", use_itn=True)
    assert len(result) == 1


def test_infer_with_hotword(client):
    """infer(hotword=...) sends hotword param."""
    result = client.infer(audio="test.wav", hotword="test hotword")
    assert len(result) == 1


def test_infer_none_generate_params_not_sent(client):
    """None generate params should not be included."""
    # This test verifies via the mock server echo — no extra params
    result = client.infer(audio="test.wav")
    assert len(result) == 1


# ------------------------------------------------------------------
# RPC API via mock server — transcribe
# ------------------------------------------------------------------

def test_transcribe(client):
    result = client.transcribe(audio="test.wav")
    assert len(result) == 1
    assert result[0]["text"] == "hello world"


def test_transcribe_with_bytes(client):
    result = client.transcribe(audio_bytes=b"fake audio data")
    assert len(result) == 1


def test_transcribe_no_input(client):
    with pytest.raises(ValueError, match="Provide exactly one of"):
        client.transcribe()


# ------------------------------------------------------------------
# RPC API via mock server — other methods
# ------------------------------------------------------------------

def test_unload_model(client):
    result = client.unload_model()
    assert result["status"] == "unloaded"


def test_unload_model_by_name(client):
    result = client.unload_model(name="my_model")
    assert result["name"] == "my_model"
    assert result["status"] == "unloaded"


def test_list_models(client):
    result = client.list_models()
    assert "models" in result


def test_execute(client):
    result = client.execute("x = 1")
    assert result["output"] == "ok"


def test_download_model_auto_hub(client):
    """download_model auto-detects hub when hub=None."""
    with patch("funasr_server.client.get_hub", return_value="hf"):
        result = client.download_model(model="SenseVoiceSmall")
    assert result["hub"] == "hf"
    assert result["model"] == "FunAudioLLM/SenseVoiceSmall"


def test_download_model_auto_hub_ms(client):
    """download_model auto-detects hub=ms in China."""
    with patch("funasr_server.client.get_hub", return_value="ms"):
        result = client.download_model(model="SenseVoiceSmall")
    assert result["hub"] == "ms"
    assert result["model"] == "iic/SenseVoiceSmall"


def test_download_model_explicit_hub(client):
    """download_model with explicit hub."""
    result = client.download_model(model="test-model", hub="hf")
    assert result["hub"] == "hf"


# ------------------------------------------------------------------
# Error handling — JSON-RPC validation
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


def test_rpc_rejects_bad_jsonrpc_version(client):
    """Response with wrong jsonrpc version raises ConnectionError."""
    with pytest.raises(ConnectionError, match="jsonrpc='2.0'"):
        client._rpc_call("bad_jsonrpc", {})


def test_rpc_rejects_mismatched_id(client):
    """Response with wrong ID raises ConnectionError."""
    with pytest.raises(ConnectionError, match="ID mismatch"):
        client._rpc_call("bad_id", {})


def test_rpc_rejects_missing_result(client):
    """Response without 'result' or 'error' raises ConnectionError."""
    with pytest.raises(ConnectionError, match="missing 'result'"):
        client._rpc_call("no_result", {})


def test_rpc_rejects_malformed_error(client):
    """Malformed error object (not a dict) raises ConnectionError."""
    with pytest.raises(ConnectionError, match="Malformed JSON-RPC error"):
        client._rpc_call("malformed_error", {})
