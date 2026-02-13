# funasr-server

Self-contained [FunASR](https://github.com/modelscope/FunASR) inference server with **one-click installation**.

No need to pre-install Python, PyTorch, or any dependencies — `funasr-server` handles everything automatically using [uv](https://docs.astral.sh/uv/).

## Features

- **Zero-config setup** — automatically installs Python, PyTorch (CPU/CUDA/MPS), and FunASR
- **Persistent server** — models stay loaded in memory, no repeated loading
- **All model types** — ASR, VAD, punctuation, speaker diarization, emotion, and more
- **Cross-platform** — Linux, macOS, Windows
- **China-friendly** — auto-detects network and uses Chinese mirrors when needed

## Quick Start

```bash
pip install funasr-server
```

```python
from funasr_server import FunASR

asr = FunASR()
asr.ensure_installed()  # one-time setup (~2 min)
asr.start()

# Load and run ASR
asr.load_model(model="iic/SenseVoiceSmall")
result = asr.infer("audio.wav", language="zh", use_itn=True)
print(result)
# [{"key": "audio", "text": "你好世界"}]

asr.stop()
```

### Context Manager

```python
with FunASR() as asr:
    asr.load_model(model="iic/SenseVoiceSmall")
    result = asr.infer("audio.wav")
```

## Multiple Model Types

```python
asr = FunASR()
asr.ensure_installed()
asr.start()

# ASR (speech recognition)
asr.load_model(model="iic/SenseVoiceSmall", name="asr")
result = asr.infer("audio.wav", name="asr")

# VAD (voice activity detection)
asr.load_model(model="fsmn-vad", name="vad")
result = asr.infer("audio.wav", name="vad")
# [{"key": "audio", "value": [[0, 3200], [4500, 9800]]}]

# Full pipeline (ASR + VAD + punctuation)
asr.load_model(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    name="pipeline",
)
result = asr.infer("audio.wav", name="pipeline")

asr.stop()
```

## API Reference

### `FunASR(runtime_dir, port, host)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `runtime_dir` | `"./funasr_runtime"` | Directory for the server environment |
| `port` | `0` (auto) | Server port |
| `host` | `"127.0.0.1"` | Bind host |

### Methods

| Method | Description |
|--------|-------------|
| `ensure_installed()` | Install runtime environment (one-time) |
| `start()` | Start the background server |
| `stop()` | Stop the server |
| `load_model(model, name, ...)` | Load any FunASR model |
| `unload_model(name)` | Unload a model and free memory |
| `infer(input, name, **kwargs)` | Run inference on any loaded model |
| `transcribe(audio, name, **kwargs)` | Convenience alias for ASR |
| `execute(code)` | Execute arbitrary Python code on the server |
| `health()` | Check server status |
| `list_models()` | List loaded models |

## Architecture

```
Your Application
    │
    │  HTTP (localhost)
    │  JSON-RPC 2.0
    ▼
FunASR Server (background process)
    │
    ├── Models loaded in memory
    ├── Isolated Python environment (uv)
    └── Auto GPU/CPU detection
```

The server runs in a completely isolated Python environment managed by `uv`. Your application communicates with it over HTTP using JSON-RPC 2.0 protocol.

## Requirements

- Python >= 3.10 (for the client SDK only)
- Internet connection (for first-time setup)
- `curl` (Linux/macOS) or PowerShell (Windows) — for auto-installing uv

## License

MIT
