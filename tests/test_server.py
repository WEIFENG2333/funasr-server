"""Tests for the server module (RPC handlers)."""

import base64
import json
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add the runtime template to path so we can import server.py
_server_path = Path(__file__).parent.parent / "src" / "funasr_server" / "runtime_template"
sys.path.insert(0, str(_server_path))

# Mock heavy dependencies before importing server
sys.modules["uvicorn"] = MagicMock()
sys.modules["starlette"] = MagicMock()
sys.modules["starlette.applications"] = MagicMock()
sys.modules["starlette.requests"] = MagicMock()
sys.modules["starlette.responses"] = MagicMock()
sys.modules["starlette.routing"] = MagicMock()

import server  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    """Reset global state between tests."""
    server._models.clear()
    server._model_kwargs.clear()
    server._exec_globals.clear()
    server._exec_globals["__builtins__"] = __builtins__
    yield
    server._models.clear()
    server._model_kwargs.clear()


# ------------------------------------------------------------------
# JSON-RPC helpers
# ------------------------------------------------------------------

def test_ok_response():
    result = server._ok(1, {"status": "ok"})
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == 1
    assert result["result"]["status"] == "ok"


def test_ok_response_null_id():
    result = server._ok(None, {"status": "ok"})
    assert result["id"] is None


def test_error_response():
    result = server._error(1, -32600, "Bad request")
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == 1
    assert result["error"]["code"] == -32600
    assert result["error"]["message"] == "Bad request"
    assert "data" not in result["error"]


def test_error_response_with_data():
    result = server._error(1, -32000, "Error", data="traceback")
    assert result["error"]["data"] == "traceback"


# ------------------------------------------------------------------
# _serialize_results
# ------------------------------------------------------------------

def test_serialize_results_list():
    result = server._serialize_results([{"key": "a", "text": "hello"}])
    assert result == [{"key": "a", "text": "hello"}]


def test_serialize_results_single_dict():
    result = server._serialize_results({"key": "a", "text": "hello"})
    assert result == [{"key": "a", "text": "hello"}]


def test_serialize_results_with_tensor():
    mock_tensor = MagicMock()
    mock_tensor.tolist.return_value = [1, 2, 3]
    result = server._serialize_results([{"embedding": mock_tensor}])
    assert result == [{"embedding": [1, 2, 3]}]


def test_serialize_results_empty_list():
    result = server._serialize_results([])
    assert result == []


def test_serialize_results_non_dict_items():
    """Non-dict items are passed through as-is."""
    result = server._serialize_results(["hello", 42])
    assert result == ["hello", 42]


def test_serialize_results_mixed():
    """Mix of dicts and non-dicts."""
    result = server._serialize_results([{"key": "a"}, "raw_string"])
    assert result == [{"key": "a"}, "raw_string"]


# ------------------------------------------------------------------
# rpc_load_model
# ------------------------------------------------------------------

def test_rpc_load_model():
    mock_model = MagicMock()
    with patch.dict("sys.modules", {"funasr": MagicMock()}):
        sys.modules["funasr"].AutoModel.return_value = mock_model
        result = server.rpc_load_model({"model": "test-model", "name": "m1"})
    assert result["name"] == "m1"
    assert result["status"] == "loaded"
    assert "m1" in server._models


def test_rpc_load_model_default_name():
    """Default name is 'default' when not specified."""
    mock_model = MagicMock()
    with patch.dict("sys.modules", {"funasr": MagicMock()}):
        sys.modules["funasr"].AutoModel.return_value = mock_model
        result = server.rpc_load_model({"model": "test-model"})
    assert result["name"] == "default"
    assert "default" in server._models


def test_rpc_load_model_already_loaded():
    mock_model = MagicMock()
    server._models["m1"] = mock_model
    server._model_kwargs["m1"] = {"model": "test"}

    with patch.dict("sys.modules", {"funasr": MagicMock()}):
        result = server.rpc_load_model({"model": "test", "name": "m1"})
    assert result["status"] == "already_loaded"


def test_rpc_load_model_replace_existing():
    """Loading a different model on same name replaces the old one."""
    old_model = MagicMock()
    server._models["m1"] = old_model
    server._model_kwargs["m1"] = {"model": "old-model"}

    new_model = MagicMock()
    with patch.dict("sys.modules", {"funasr": MagicMock()}):
        sys.modules["funasr"].AutoModel.return_value = new_model
        result = server.rpc_load_model({"model": "new-model", "name": "m1"})

    assert result["status"] == "loaded"
    assert server._models["m1"] is new_model
    assert server._model_kwargs["m1"]["model"] == "new-model"


def test_rpc_load_model_passes_kwargs():
    """Extra parameters are passed to AutoModel."""
    with patch.dict("sys.modules", {"funasr": MagicMock()}) as mock_funasr:
        sys.modules["funasr"].AutoModel.return_value = MagicMock()
        server.rpc_load_model({"model": "test", "name": "m1", "hub": "hf", "device": "cpu"})
        sys.modules["funasr"].AutoModel.assert_called_once_with(model="test", hub="hf", device="cpu")


# ------------------------------------------------------------------
# rpc_unload_model
# ------------------------------------------------------------------

def test_rpc_unload_model():
    server._models["m1"] = MagicMock()
    server._model_kwargs["m1"] = {"model": "test"}

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = server.rpc_unload_model({"name": "m1"})

    assert result["status"] == "unloaded"
    assert "m1" not in server._models
    assert "m1" not in server._model_kwargs


def test_rpc_unload_model_not_found():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = server.rpc_unload_model({"name": "nonexistent"})
    assert result["status"] == "not_found"


def test_rpc_unload_model_clears_cuda_cache():
    """CUDA cache is cleared when CUDA is available."""
    server._models["m1"] = MagicMock()
    server._model_kwargs["m1"] = {"model": "test"}

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        server.rpc_unload_model({"name": "m1"})

    mock_torch.cuda.empty_cache.assert_called_once()


# ------------------------------------------------------------------
# rpc_infer
# ------------------------------------------------------------------

def test_rpc_infer():
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"key": "test", "text": "hello"}]
    server._models["default"] = mock_model

    result = server.rpc_infer({"input": "test.wav"})
    assert result["results"] == [{"key": "test", "text": "hello"}]
    mock_model.generate.assert_called_once_with(input="test.wav")


def test_rpc_infer_by_name():
    """Inference uses the model specified by name."""
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"key": "test", "value": [[0, 1000]]}]
    server._models["vad"] = mock_model

    result = server.rpc_infer({"input": "test.wav", "name": "vad"})
    assert result["results"] == [{"key": "test", "value": [[0, 1000]]}]


def test_rpc_infer_passes_extra_kwargs():
    """Extra params are forwarded to model.generate()."""
    mock_model = MagicMock()
    mock_model.generate.return_value = []
    server._models["default"] = mock_model

    server.rpc_infer({"input": "test.wav", "language": "zh", "use_itn": True})
    mock_model.generate.assert_called_once_with(input="test.wav", language="zh", use_itn=True)


def test_rpc_infer_model_not_loaded():
    with pytest.raises(ValueError, match="not loaded"):
        server.rpc_infer({"input": "test.wav"})


def test_rpc_infer_no_input():
    server._models["default"] = MagicMock()
    with pytest.raises(ValueError, match="input"):
        server.rpc_infer({})


def test_rpc_infer_with_base64():
    """input_base64 is decoded and written to a temp file."""
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"key": "test", "text": "hello"}]
    server._models["default"] = mock_model

    audio_bytes = b"fake audio content"
    encoded = base64.b64encode(audio_bytes).decode()

    result = server.rpc_infer({"input_base64": encoded})
    assert result["results"] == [{"key": "test", "text": "hello"}]

    # Verify generate was called with a file path (temp file)
    call_args = mock_model.generate.call_args
    input_path = call_args[1]["input"]
    assert isinstance(input_path, str)


# ------------------------------------------------------------------
# rpc_transcribe
# ------------------------------------------------------------------

def test_rpc_transcribe_maps_params():
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"key": "test", "text": "hello"}]
    server._models["default"] = mock_model

    result = server.rpc_transcribe({"audio": "test.wav"})
    assert result["results"] == [{"key": "test", "text": "hello"}]
    mock_model.generate.assert_called_once_with(input="test.wav")


def test_rpc_transcribe_maps_audio_base64():
    """audio_base64 is mapped to input_base64."""
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"key": "test", "text": "hello"}]
    server._models["default"] = mock_model

    encoded = base64.b64encode(b"fake audio").decode()
    result = server.rpc_transcribe({"audio_base64": encoded})
    assert result["results"] == [{"key": "test", "text": "hello"}]


# ------------------------------------------------------------------
# rpc_execute
# ------------------------------------------------------------------

def test_rpc_execute():
    result = server.rpc_execute({"code": "result = 42"})
    assert result["return_value"] == 42


def test_rpc_execute_with_output():
    result = server.rpc_execute({"code": "print('hello')"})
    assert "hello" in result["output"]


def test_rpc_execute_error():
    result = server.rpc_execute({"code": "raise ValueError('boom')"})
    assert "error" in result
    assert "boom" in result["error"]


def test_rpc_execute_globals_persist():
    """Variables set in one execute call persist to the next."""
    server.rpc_execute({"code": "my_var = 123"})
    result = server.rpc_execute({"code": "result = my_var + 1"})
    assert result["return_value"] == 124


def test_rpc_execute_custom_return_var():
    """Custom return_var parameter works."""
    result = server.rpc_execute({"code": "answer = 99", "return_var": "answer"})
    assert result["return_value"] == 99


def test_rpc_execute_no_result():
    """No return_value when 'result' variable is not set."""
    result = server.rpc_execute({"code": "x = 1"})
    assert result["return_value"] is None
    assert result.get("error") is None


def test_rpc_execute_non_serializable_return():
    """Non-JSON-serializable return values are converted to string."""
    result = server.rpc_execute({"code": "import os; result = os"})
    assert isinstance(result["return_value"], str)


# ------------------------------------------------------------------
# rpc_download_model
# ------------------------------------------------------------------

def test_rpc_download_model_ms():
    """Download model from ModelScope."""
    with patch.dict("sys.modules", {"modelscope": MagicMock(), "modelscope.hub": MagicMock(), "modelscope.hub.snapshot_download": MagicMock()}):
        sys.modules["modelscope.hub.snapshot_download"].snapshot_download.return_value = "/tmp/model"
        result = server.rpc_download_model({"model": "iic/SenseVoiceSmall", "hub": "ms"})

    assert result["model"] == "iic/SenseVoiceSmall"
    assert result["path"] == "/tmp/model"
    assert result["hub"] == "ms"


def test_rpc_download_model_hf():
    """Download model from HuggingFace."""
    with patch.dict("sys.modules", {"huggingface_hub": MagicMock()}):
        sys.modules["huggingface_hub"].snapshot_download.return_value = "/tmp/model_hf"
        result = server.rpc_download_model({"model": "FunAudioLLM/SenseVoiceSmall", "hub": "hf"})

    assert result["model"] == "FunAudioLLM/SenseVoiceSmall"
    assert result["path"] == "/tmp/model_hf"
    assert result["hub"] == "hf"


def test_rpc_download_model_no_model():
    """Missing model parameter raises ValueError."""
    with pytest.raises(ValueError, match="model"):
        server.rpc_download_model({"hub": "ms"})


def test_rpc_download_model_unknown_hub():
    """Unknown hub raises ValueError."""
    with pytest.raises(ValueError, match="Unknown hub"):
        server.rpc_download_model({"model": "test", "hub": "unknown"})


# ------------------------------------------------------------------
# rpc_list_models
# ------------------------------------------------------------------

def test_rpc_list_models_empty():
    result = server.rpc_list_models({})
    assert result["models"] == {}


def test_rpc_list_models_with_models():
    server._models["m1"] = MagicMock()
    server._model_kwargs["m1"] = {"model": "test"}

    result = server.rpc_list_models({})
    assert "m1" in result["models"]
    assert result["models"]["m1"]["kwargs"] == {"model": "test"}


def test_rpc_list_models_multiple():
    """Multiple models appear in list."""
    server._models["vad"] = MagicMock()
    server._models["asr"] = MagicMock()
    server._model_kwargs["vad"] = {"model": "fsmn-vad"}
    server._model_kwargs["asr"] = {"model": "SenseVoice"}

    result = server.rpc_list_models({})
    assert len(result["models"]) == 2
    assert "vad" in result["models"]
    assert "asr" in result["models"]


# ------------------------------------------------------------------
# Dispatch table
# ------------------------------------------------------------------

def test_methods_dispatch_table():
    expected = {"health", "load_model", "unload_model", "infer", "transcribe",
                "execute", "download_model", "list_models", "shutdown"}
    assert expected == set(server._METHODS.keys())
