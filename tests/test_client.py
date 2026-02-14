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

from funasr_server.client import FunASR, Model, ServerError


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
            result = dict(params)
            result["status"] = "loaded"
        elif method == "unload_model":
            result = {"name": params.get("name"), "status": "unloaded"}
        elif method == "infer":
            result = {"results": [{"key": "test", "text": "hello world"}], "params_echo": params}
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
        pass


@pytest.fixture
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockRPCHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


@pytest.fixture
def client(mock_server):
    c = FunASR(port=mock_server, host="127.0.0.1")
    c._process = None
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
        assert c._model_counter == 0


def test_is_running_no_process():
    c = FunASR()
    assert c.is_running() is False


def test_is_running_with_live_process():
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    c._process = mock_proc
    assert c.is_running() is True


def test_is_running_with_dead_process():
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1
    c._process = mock_proc
    assert c.is_running() is False


def test_stop_no_process():
    c = FunASR()
    c.stop()


def test_stop_graceful():
    c = FunASR()
    mock_proc = MagicMock()
    c._process = mock_proc
    with patch.object(c, "_rpc_call"):
        c.stop()
    mock_proc.wait.assert_called_once()
    assert c._process is None


def test_stop_force_kill_windows():
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
    c._process = mock_proc
    with patch.object(c, "_rpc_call"), \
         patch("platform.system", return_value="Windows"):
        c.stop()
    mock_proc.terminate.assert_called_once()
    assert c._process is None


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="SIGKILL only on Unix")
def test_stop_force_kill_unix():
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), None]
    c._process = mock_proc
    with patch.object(c, "_rpc_call"), \
         patch("platform.system", return_value="Linux"):
        c.stop()
    mock_proc.send_signal.assert_called_once_with(signal.SIGKILL)
    assert c._process is None


def test_ensure_installed_already():
    c = FunASR()
    with patch.object(c.installer, "is_installed", return_value=True):
        assert c.ensure_installed() is True


def test_ensure_installed_fresh():
    c = FunASR()
    with patch.object(c.installer, "is_installed", return_value=False), \
         patch.object(c.installer, "install"):
        assert c.ensure_installed() is False


def test_start_no_uv():
    c = FunASR()
    with patch.object(c.installer, "get_uv_path", return_value=None):
        with pytest.raises(RuntimeError, match="uv not found"):
            c.start()


def test_start_already_running():
    c = FunASR()
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    c._process = mock_proc
    c.port = 1234
    assert c.start() == 1234


# ------------------------------------------------------------------
# load_model → Model
# ------------------------------------------------------------------

def test_load_model_returns_model(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    assert isinstance(model, Model)


def test_load_model_auto_name(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        m1 = client.load_model("test-model")
        m2 = client.load_model("test-model")
    assert m1.name == "model_1"
    assert m2.name == "model_2"


def test_load_model_explicit_name(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model", name="my_asr")
    assert model.name == "my_asr"


def test_load_model_resolves_hf(client):
    model = client.load_model("SenseVoiceSmall", hub="hf")
    assert isinstance(model, Model)


def test_load_model_with_pipeline(client):
    model = client.load_model(
        "SenseVoiceSmall",
        vad_model="fsmn-vad", punc_model="ct-punc", spk_model="cam++",
        hub="ms",
    )
    assert isinstance(model, Model)


def test_load_model_with_options(client):
    model = client.load_model(
        "test-model", hub="ms",
        batch_size=4, fp16=True, quantize=True, disable_update=True, device="cpu",
    )
    assert isinstance(model, Model)


def test_load_model_kwargs(client):
    model = client.load_model("test-model", hub="ms", trust_remote_code=True, ncpu=4)
    assert isinstance(model, Model)


# ------------------------------------------------------------------
# Model — infer / call / transcribe / unload
# ------------------------------------------------------------------

def test_model_infer_audio(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    result = model.infer(audio="test.wav")
    assert result[0]["text"] == "hello world"


def test_model_infer_text(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("ct-punc")
    result = model.infer(text="你好世界")
    assert len(result) == 1


def test_model_infer_audio_bytes(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    result = model.infer(audio_bytes=b"fake audio")
    assert len(result) == 1


def test_model_infer_no_input(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    with pytest.raises(ValueError, match="Provide exactly one of"):
        model.infer()


def test_model_infer_with_kwargs(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    result = model.infer(audio="test.wav", language="zh", use_itn=True, hotword="test")
    assert len(result) == 1


def test_model_call(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    result = model(audio="test.wav")
    assert result[0]["text"] == "hello world"


def test_model_transcribe(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    result = model.transcribe(audio="test.wav")
    assert result[0]["text"] == "hello world"


def test_model_transcribe_bytes(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model")
    result = model.transcribe(audio_bytes=b"fake audio")
    assert len(result) == 1


def test_model_unload(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model", name="my_model")
    result = model.unload()
    assert result["status"] == "unloaded"
    assert result["name"] == "my_model"


def test_model_repr(client):
    with patch("funasr_server.client.get_hub", return_value="ms"):
        model = client.load_model("test-model", name="asr")
    assert repr(model) == "Model(name='asr')"


# ------------------------------------------------------------------
# Other methods
# ------------------------------------------------------------------

def test_health(client):
    result = client.health()
    assert result["status"] == "ok"


def test_list_models(client):
    result = client.list_models()
    assert "models" in result


def test_execute(client):
    result = client.execute("x = 1")
    assert result["output"] == "ok"


def test_download_model_auto_hub(client):
    with patch("funasr_server.client.get_hub", return_value="hf"):
        result = client.download_model(model="SenseVoiceSmall")
    assert result["hub"] == "hf"
    assert result["model"] == "FunAudioLLM/SenseVoiceSmall"


def test_download_model_explicit_hub(client):
    result = client.download_model(model="test-model", hub="hf")
    assert result["hub"] == "hf"


# ------------------------------------------------------------------
# JSON-RPC validation
# ------------------------------------------------------------------

def test_server_error(client):
    with pytest.raises(ServerError, match="test error"):
        client._rpc_call("error_test", {})


def test_server_error_with_data(client):
    with pytest.raises(ServerError) as exc_info:
        client._rpc_call("error_with_data", {})
    assert exc_info.value.code == -32000
    assert exc_info.value.data == "traceback info"


def test_server_error_attributes():
    err = ServerError(-32000, "test message", "extra data")
    assert err.code == -32000
    assert err.data == "extra data"
    assert str(err) == "test message"


def test_connection_error():
    c = FunASR(port=1, host="127.0.0.1")
    with pytest.raises(ConnectionError):
        c._rpc_call("health", {}, timeout=1)


def test_rpc_id_increments(client):
    initial = client._rpc_id
    client.health()
    assert client._rpc_id == initial + 1
    client.health()
    assert client._rpc_id == initial + 2


def test_rpc_rejects_bad_jsonrpc(client):
    with pytest.raises(ConnectionError, match="jsonrpc='2.0'"):
        client._rpc_call("bad_jsonrpc", {})


def test_rpc_rejects_bad_id(client):
    with pytest.raises(ConnectionError, match="ID mismatch"):
        client._rpc_call("bad_id", {})


def test_rpc_rejects_no_result(client):
    with pytest.raises(ConnectionError, match="missing 'result'"):
        client._rpc_call("no_result", {})


def test_rpc_rejects_malformed_error(client):
    with pytest.raises(ConnectionError, match="Malformed JSON-RPC error"):
        client._rpc_call("malformed_error", {})
