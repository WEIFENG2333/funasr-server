"""Tests for the server module (RPC handlers)."""

import json
import sys
import os
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


def test_ok_response():
    result = server._ok(1, {"status": "ok"})
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == 1
    assert result["result"]["status"] == "ok"


def test_error_response():
    result = server._error(1, -32600, "Bad request")
    assert result["jsonrpc"] == "2.0"
    assert result["id"] == 1
    assert result["error"]["code"] == -32600
    assert result["error"]["message"] == "Bad request"


def test_error_response_with_data():
    result = server._error(1, -32000, "Error", data="traceback")
    assert result["error"]["data"] == "traceback"


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


def test_rpc_load_model():
    mock_model = MagicMock()
    with patch.dict("sys.modules", {"funasr": MagicMock()}):
        sys.modules["funasr"].AutoModel.return_value = mock_model
        result = server.rpc_load_model({"model": "test-model", "name": "m1"})
    assert result["name"] == "m1"
    assert result["status"] == "loaded"
    assert "m1" in server._models


def test_rpc_load_model_already_loaded():
    mock_model = MagicMock()
    server._models["m1"] = mock_model
    server._model_kwargs["m1"] = {"model": "test"}

    with patch.dict("sys.modules", {"funasr": MagicMock()}):
        result = server.rpc_load_model({"model": "test", "name": "m1"})
    assert result["status"] == "already_loaded"


def test_rpc_unload_model():
    server._models["m1"] = MagicMock()
    server._model_kwargs["m1"] = {"model": "test"}

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = server.rpc_unload_model({"name": "m1"})

    assert result["status"] == "unloaded"
    assert "m1" not in server._models


def test_rpc_unload_model_not_found():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        result = server.rpc_unload_model({"name": "nonexistent"})
    assert result["status"] == "not_found"


def test_rpc_infer():
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"key": "test", "text": "hello"}]
    server._models["default"] = mock_model

    result = server.rpc_infer({"input": "test.wav"})
    assert result["results"] == [{"key": "test", "text": "hello"}]
    mock_model.generate.assert_called_once_with(input="test.wav")


def test_rpc_infer_model_not_loaded():
    with pytest.raises(ValueError, match="not loaded"):
        server.rpc_infer({"input": "test.wav"})


def test_rpc_infer_no_input():
    server._models["default"] = MagicMock()
    with pytest.raises(ValueError, match="input"):
        server.rpc_infer({})


def test_rpc_transcribe_maps_params():
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"key": "test", "text": "hello"}]
    server._models["default"] = mock_model

    result = server.rpc_transcribe({"audio": "test.wav"})
    assert result["results"] == [{"key": "test", "text": "hello"}]
    mock_model.generate.assert_called_once_with(input="test.wav")


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


def test_rpc_list_models_empty():
    result = server.rpc_list_models({})
    assert result["models"] == {}


def test_rpc_list_models_with_models():
    server._models["m1"] = MagicMock()
    server._model_kwargs["m1"] = {"model": "test"}

    result = server.rpc_list_models({})
    assert "m1" in result["models"]
    assert result["models"]["m1"]["kwargs"] == {"model": "test"}


def test_methods_dispatch_table():
    assert "health" in server._METHODS
    assert "load_model" in server._METHODS
    assert "unload_model" in server._METHODS
    assert "infer" in server._METHODS
    assert "transcribe" in server._METHODS
    assert "execute" in server._METHODS
    assert "download_model" in server._METHODS
    assert "list_models" in server._METHODS
    assert "shutdown" in server._METHODS
