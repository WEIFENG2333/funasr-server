# CLAUDE.md

This file provides guidance to Claude Code when working on this project.

## Project Overview

funasr-server is a self-contained FunASR inference server with one-click installation. It provides a Python client SDK that communicates with a background server process over HTTP using JSON-RPC 2.0.

## Architecture

```
Host Application (Python 3.10+)
    │  HTTP + JSON-RPC 2.0
    ▼
FunASR Server (Starlette + uvicorn, isolated uv environment)
    │  Models cached in memory
    ▼
FunASR Core (PyTorch)
```

- **Client SDK** (`src/funasr_server/client.py`): `FunASR` class manages server lifecycle, `Model` class wraps loaded models
- **Server** (`src/funasr_server/runtime_template/server.py`): Runs in an isolated Python environment managed by uv. Copied to runtime dir during installation
- **Model Registry** (`src/funasr_server/models.py`): 40+ models with hub-agnostic name resolution (ModelScope for China, HuggingFace internationally)
- **Installer** (`src/funasr_server/installer.py`): Handles uv installation, runtime directory creation, dependency sync
- **Mirror** (`src/funasr_server/mirror.py`): Region detection (China vs international) via concurrent host probing

## Key Files

| File | Purpose |
|------|---------|
| `src/funasr_server/__init__.py` | Public API exports |
| `src/funasr_server/client.py` | Core client SDK (FunASR, Model, ServerError) |
| `src/funasr_server/installer.py` | Runtime environment setup |
| `src/funasr_server/mirror.py` | Region detection and mirror selection |
| `src/funasr_server/models.py` | Model registry and name resolution |
| `src/funasr_server/runtime_template/server.py` | JSON-RPC server (deployed to runtime dir) |
| `src/funasr_server/runtime_template/pyproject.toml` | Server-side dependencies |

## Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run unit tests only (no real server/models)
python -m pytest tests/ -v -m "not integration"

# Run integration tests (requires GPU/models, slow)
python -m pytest tests/ -v -m integration

# Run a single test file
python -m pytest tests/test_client.py -v
```

## Testing

- **Unit tests** (`test_client.py`, `test_server.py`, `test_installer.py`, `test_mirror.py`, `test_models.py`): Use mocks, no real server needed
- **Integration tests** (`test_integration.py`): Start real server, load models, run inference. Marked with `@pytest.mark.integration`
- Test audio fixture: `tests/fixtures/asr_example.wav`
- Mock RPC handler in `test_client.py` simulates server responses

## Important Patterns

- **Runtime template**: `server.py` is copied from `runtime_template/` to the runtime directory during `Installer.install()`. Changes to `runtime_template/server.py` require either reinstalling or manually copying to the runtime dir for testing
- **Model name resolution**: User-friendly names (e.g. "SenseVoiceSmall") are resolved to hub-specific IDs (e.g. "iic/SenseVoiceSmall" for ModelScope) via `resolve_model_id()`
- **Blocking methods**: `infer`, `transcribe`, `load_model`, `download_model` run in `run_in_executor` on the server so concurrent requests (like `get_progress`) can be handled
- **Progress tracking**: Server stores progress in `_progress` dict during inference, client polls via `get_progress` RPC. Progress granularity depends on VAD segment count

## Dependencies

- **Client SDK**: Zero external dependencies (pure Python 3.10+)
- **Server runtime**: FunASR, PyTorch 2.0+, torchaudio, transformers, uvicorn, starlette (managed by uv in isolated environment)
- **Dev**: pytest >= 8.0

## Pipeline Compatibility

- `paraformer-zh` + `fsmn-vad` + `ct-punc`: Full pipeline, official recommendation
- `SenseVoiceSmall` + `fsmn-vad`: OK. Do NOT combine with `ct-punc` (corrupts special tags)
- `Fun-ASR-Nano`: Standalone only. Errors with VAD (batch_size incompatibility in FunASR framework)
- `cam++`, `emotion2vec_*`, `ct-punc`, `fsmn-vad`: Standalone only
