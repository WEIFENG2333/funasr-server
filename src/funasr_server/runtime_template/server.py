"""FunASR Server — JSON-RPC 2.0 over HTTP.

A long-running process that keeps models loaded in memory and accepts
JSON-RPC requests from the host application.

Usage:
    python server.py --port 9520
    python server.py --port 0    # auto-assign port, print to stdout
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import tempfile
import traceback

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger("funasr_server")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_models: dict = {}  # name -> AutoModel instance
_model_kwargs: dict = {}  # name -> kwargs used to create it
_exec_globals: dict = {"__builtins__": __builtins__}  # shared exec namespace


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _ok(id, result):
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": err}


# ---------------------------------------------------------------------------
# RPC method implementations
# ---------------------------------------------------------------------------

def rpc_health(params: dict) -> dict:
    """Health check. Returns server status."""
    import torch
    return {
        "status": "ok",
        "loaded_models": list(_models.keys()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
    }


def rpc_load_model(params: dict) -> dict:
    """Load an AutoModel instance and cache it.

    Params:
        name (str): Cache key for this model (default: "default")
        model (str): Model name/path, e.g. "iic/SenseVoiceSmall"
        vad_model (str, optional): VAD model name
        punc_model (str, optional): Punctuation model name
        spk_model (str, optional): Speaker model name
        device (str, optional): "cuda" / "cpu" / "auto"
        hub (str, optional): "ms" / "hf"
        batch_size (int, optional): Inference batch size
        quantize (bool, optional): Enable quantization
        fp16 (bool, optional): Enable half-precision
        disable_update (bool, optional): Skip model update checks
        **remaining: Any additional AutoModel parameters
    """
    from funasr import AutoModel

    name = params.get("name", "default")
    model_kwargs = {k: v for k, v in params.items() if k != "name"}

    if name in _models and _model_kwargs.get(name) == model_kwargs:
        return {"name": name, "status": "already_loaded"}

    if name in _models:
        del _models[name]
        _model_kwargs.pop(name, None)

    logger.info(f"Loading model '{name}': {model_kwargs}")
    model = AutoModel(**model_kwargs)
    _models[name] = model
    _model_kwargs[name] = model_kwargs

    return {"name": name, "status": "loaded"}


def rpc_unload_model(params: dict) -> dict:
    """Unload a model and free memory.

    Params:
        name (str): Model cache key (default: "default")
    """
    import torch

    name = params.get("name", "default")
    if name in _models:
        del _models[name]
        _model_kwargs.pop(name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"name": name, "status": "unloaded"}
    return {"name": name, "status": "not_found"}


def _serialize_results(result) -> list:
    """Convert model output to JSON-serializable format."""
    if not isinstance(result, (list, tuple)):
        result = [result]
    serializable = []
    for item in result:
        if isinstance(item, dict):
            entry = {}
            for k, v in item.items():
                if hasattr(v, "tolist"):  # torch tensor / numpy array
                    entry[k] = v.tolist()
                else:
                    entry[k] = v
            serializable.append(entry)
        else:
            serializable.append(item)
    return serializable


def rpc_infer(params: dict) -> dict:
    """Universal inference — works with ANY FunASR model.

    Calls model.generate(input=..., **kwargs) on the specified model.

    Params:
        name (str): Model cache key (default: "default")
        input (str): Input data — file path (audio) or text string
        input_base64 (str): Base64-encoded audio bytes (alternative to input)
        audio_format (str): Suffix for temp file when using input_base64 (default: ".wav")
        language (str, optional): Language code (e.g. "zh", "en")
        use_itn (bool, optional): Enable inverse text normalization
        batch_size (int, optional): Inference batch size
        hotword (str, optional): Hotword file path or string
        merge_vad (bool, optional): Merge short VAD segments
        merge_length_s (float, optional): Max merge length in seconds
        output_timestamp (bool, optional): Include timestamps in output
        **remaining: Any additional generate() parameters
    """
    name = params.get("name", "default")
    if name not in _models:
        raise ValueError(f"Model '{name}' not loaded. Call load_model first.")

    model = _models[name]

    input_data = params.get("input")
    input_base64 = params.get("input_base64")
    tmp_file = None

    # Build generate kwargs: everything except control params
    _control_keys = {"name", "input", "input_base64", "audio_format"}
    generate_kwargs = {k: v for k, v in params.items() if k not in _control_keys}

    if input_base64:
        audio_bytes = base64.b64decode(input_base64)
        suffix = params.get("audio_format", ".wav")
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(audio_bytes)
        tmp.close()
        input_data = tmp.name
        tmp_file = tmp.name

    if input_data is None:
        raise ValueError("'input' or 'input_base64' is required")

    try:
        result = model.generate(input=input_data, **generate_kwargs)
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)

    return {"results": _serialize_results(result)}


def rpc_transcribe(params: dict) -> dict:
    """Transcribe audio — convenience alias for infer().

    Same as infer(), but uses 'audio'/'audio_base64' param names
    for backward compatibility.
    """
    # Remap param names without mutating the original dict
    mapped = dict(params)
    if "audio" in mapped:
        mapped["input"] = mapped.pop("audio")
    if "audio_base64" in mapped:
        mapped["input_base64"] = mapped.pop("audio_base64")
    return rpc_infer(mapped)


def rpc_execute(params: dict) -> dict:
    """Execute arbitrary Python code in the server environment.

    Params:
        code (str): Python code to execute
        return_var (str, optional): Variable name whose value to return

    Returns:
        {"output": stdout_capture, "return_value": value_of_return_var}
    """
    code = params.get("code", "")
    return_var = params.get("return_var", "result")

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        exec(code, _exec_globals)
    except Exception:
        sys.stdout = old_stdout
        return {
            "output": captured.getvalue(),
            "error": traceback.format_exc(),
        }

    sys.stdout = old_stdout
    output = captured.getvalue()

    return_value = _exec_globals.get(return_var)
    if return_value is not None:
        try:
            json.dumps(return_value)
        except (TypeError, ValueError):
            return_value = str(return_value)

    return {
        "output": output,
        "return_value": return_value,
    }


def rpc_download_model(params: dict) -> dict:
    """Download a model to local cache.

    Params:
        model (str): Model ID, e.g. "iic/SenseVoiceSmall"
        hub (str): "ms" or "hf" (default: "ms")
    """
    model_id = params.get("model")
    hub = params.get("hub", "ms")

    if not model_id:
        raise ValueError("'model' parameter is required")

    if hub in ("ms", "modelscope"):
        from modelscope.hub.snapshot_download import snapshot_download
        path = snapshot_download(model_id)
    elif hub in ("hf", "huggingface"):
        from huggingface_hub import snapshot_download
        path = snapshot_download(model_id)
    else:
        raise ValueError(f"Unknown hub: {hub}")

    return {"model": model_id, "path": str(path), "hub": hub}


def rpc_list_models(params: dict) -> dict:
    """List all loaded models."""
    return {
        "models": {
            name: {"kwargs": _model_kwargs.get(name, {})}
            for name in _models
        }
    }


def rpc_shutdown(params: dict) -> dict:
    """Gracefully shut down the server."""
    logger.info("Shutdown requested")
    os._exit(0)


# Method dispatch table
_METHODS = {
    "health": rpc_health,
    "load_model": rpc_load_model,
    "unload_model": rpc_unload_model,
    "infer": rpc_infer,
    "transcribe": rpc_transcribe,
    "execute": rpc_execute,
    "download_model": rpc_download_model,
    "list_models": rpc_list_models,
    "shutdown": rpc_shutdown,
}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

async def handle_rpc(request: Request):
    """Handle JSON-RPC 2.0 requests."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(_error(None, -32700, "Parse error"))

    if not isinstance(body, dict):
        return JSONResponse(_error(None, -32600, "Invalid Request: expected JSON object"))

    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    if not isinstance(method, str) or not method:
        return JSONResponse(_error(req_id, -32600, "Invalid Request: 'method' must be a non-empty string"))

    if not isinstance(params, dict):
        return JSONResponse(_error(req_id, -32602, "Invalid params: expected JSON object"))

    handler = _METHODS.get(method)
    if not handler:
        return JSONResponse(_error(req_id, -32601, f"Method not found: {method}"))

    try:
        result = handler(params)
        return JSONResponse(_ok(req_id, result))
    except Exception as e:
        logger.exception(f"Error in method '{method}'")
        return JSONResponse(_error(req_id, -32000, str(e), data=traceback.format_exc()))


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Starlette(
    routes=[
        Route("/rpc", handle_rpc, methods=["POST"]),
    ],
)


def main():
    parser = argparse.ArgumentParser(description="FunASR Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9520, help="Bind port (default: 9520, 0=auto)")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.port == 0:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((args.host, 0))
        args.port = sock.getsockname()[1]
        sock.close()

    print(f"PLUGIN_PORT={args.port}", flush=True)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
