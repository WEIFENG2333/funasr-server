# funasr-server

Self-contained [FunASR](https://github.com/modelscope/FunASR) inference server with **one-click installation**.

No need to pre-install Python, PyTorch, or any dependencies — `funasr-server` handles everything automatically using [uv](https://docs.astral.sh/uv/).

## Features

- **Zero-config setup** — automatically installs Python, PyTorch (CPU/CUDA/MPS), and FunASR
- **Persistent server** — models stay loaded in memory, no repeated loading
- **All model types** — ASR, VAD, punctuation, speaker embedding, emotion recognition
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

# Load model — returns a Model handle
model = asr.load_model("SenseVoiceSmall")

# Run inference
result = model.infer(audio="audio.wav")
print(result)
# [{"key": "audio", "text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>你好世界"}]

# Or use shorthand
result = model("audio.wav")

model.unload()
asr.stop()
```

### Context Manager

```python
with FunASR() as asr:
    model = asr.load_model("SenseVoiceSmall")
    result = model("audio.wav")
```

## Supported Models

### ASR (Speech Recognition)

#### SenseVoiceSmall

Multi-task ASR with language/emotion/event detection. 234M params, supports zh/en/ja/ko/yue.

```python
model = asr.load_model("SenseVoiceSmall")
result = model(audio="audio.wav")
```

Output:
```python
[{"key": "audio", "text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验达摩院推出的语音识别模型"}]
```

The text field contains special tags: `<|language|><|emotion|><|event|><|itn|>text`.

**Inference parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `language` | `str` | Language hint: `"zh"`, `"en"`, `"ja"`, `"ko"`, `"yue"` |
| `use_itn` | `bool` | Enable inverse text normalization (adds punctuation, tag changes to `<\|withitn\|>`) |
| `batch_size` | `int` | Batch size for processing multiple files |

```python
# With ITN enabled — adds punctuation
model = asr.load_model("SenseVoiceSmall")
result = model(audio="audio.wav", use_itn=True)
# [{"key": "audio", "text": "<|zh|><|NEUTRAL|><|Speech|><|withitn|>欢迎大家来体验达摩院推出的语音识别模型。"}]
```

> **Note:** SenseVoiceSmall can be combined with `vad_model="fsmn-vad"` to process long audio. Do NOT combine with `punc_model="ct-punc"` — the punctuation model will corrupt the special tags in the output.

#### Fun-ASR-Nano

End-to-end ASR with built-in punctuation and timestamps. 800M params, supports zh (7 dialects, 26 accents) + en + ja.

```python
nano = asr.load_model("Fun-ASR-Nano")
result = nano(audio="audio.wav")
```

Output:
```python
[{
    "key": "audio",
    "text": "欢迎大家来体验达摩院推出的语音识别模型。",     # with punctuation
    "text_tn": "欢迎大家来体验达摩院推出的语音识别模型",    # without punctuation
    "timestamps": [
        {"token": "欢", "start_time": 0.0, "end_time": 3.06},
        {"token": "迎", "start_time": 3.06, "end_time": 3.12},
        ...
    ]
}]
```

> **Note:** Fun-ASR-Nano is a standalone model. Do NOT combine with `vad_model` or `punc_model`. Fun-ASR-Nano uses autoregressive decoding (token-by-token generation, like GPT), which only supports `batch_size=1`. However, FunASR's VAD pipeline (`inference_with_vad`) automatically sets a large batch size (default 300s worth of audio per batch) to process multiple VAD segments in parallel — this triggers Fun-ASR-Nano's `batch decoding is not implemented` error. This is a FunASR framework limitation, not a fundamental model constraint. Fun-ASR-Nano handles long audio end-to-end internally and does not need external VAD.

#### paraformer / paraformer-zh

Classic Paraformer ASR. 220M params. `paraformer` is for short audio (max 20s), `paraformer-zh` supports arbitrary length with SeACo.

```python
model = asr.load_model("paraformer")
result = model(audio="audio.wav")
# [{"key": "audio", "text": "欢迎大家来体验达摩院推出的语音识别模型"}]
```

`paraformer-zh` is designed for the full pipeline:

```python
model = asr.load_model("paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
result = model(audio="long_audio.wav")
# [{"key": "audio", "text": "欢迎大家来体验达摩院推出的语音识别模型。"}]
```

### VAD (Voice Activity Detection)

#### fsmn-vad

Detects speech segments in audio. 0.4M params, 16kHz.

```python
vad = asr.load_model("fsmn-vad")
result = vad(audio="audio.wav")
```

Output:
```python
[{"key": "audio", "value": [[610, 5530]]}]
```

`value` contains a list of `[start_ms, end_ms]` pairs indicating speech segments.

### Punctuation

#### ct-punc

Adds punctuation to raw text. 1.1G params, supports zh + en.

```python
punc = asr.load_model("ct-punc")
result = punc(text="你好世界今天天气真好我们一起出去玩吧")
```

Output:
```python
[{"key": "...", "text": "你好，世界今天天气真好，我们一起出去玩吧。", "punc_array": [1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 3]}]
```

`punc_array` values: `1` = none, `2` = comma, `3` = period.

### Speaker Embedding

#### cam++

Extracts speaker embedding vectors. 7.2M params, outputs 192-dim vector.

```python
spk = asr.load_model("cam++")
result = spk(audio="audio.wav")
```

Output:
```python
[{"spk_embedding": [[-0.769, 0.930, -0.338, ..., 1.158, 0.615]]}]  # 192-dim
```

Can be used for speaker verification by comparing cosine similarity between embeddings.

### Emotion Recognition

#### emotion2vec_plus_base / emotion2vec_plus_large

Speech emotion recognition. Classifies into 9 emotion categories.

```python
emo = asr.load_model("emotion2vec_plus_base")
result = emo(audio="audio.wav")
```

Output:
```python
[{
    "key": "audio",
    "labels": ["生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy",
               "中立/neutral", "其他/other", "难过/sad", "吃惊/surprised", "<unk>"],
    "scores": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "feats": [...]  # 768-dim embedding
}]
```

## Pipeline Combinations

Some models can be combined into a pipeline via `load_model()` parameters:

| Main Model | + vad_model | + punc_model | + spk_model | Notes |
|-----------|-------------|--------------|-------------|-------|
| `SenseVoiceSmall` | `fsmn-vad` | -- | -- | VAD for long audio. Do NOT use ct-punc (corrupts tags). |
| `paraformer-zh` | `fsmn-vad` | `ct-punc` | `cam++` | Full pipeline, official FunASR recommendation. |
| `paraformer-en-spk` | `fsmn-vad` | `ct-punc` | -- | English ASR with built-in speaker diarization. |
| `Fun-ASR-Nano` | -- | -- | -- | Standalone only. Errors if combined with VAD/punc. |
| `emotion2vec_*` | -- | -- | -- | Standalone only. |
| `cam++` | -- | -- | -- | Standalone only. |
| `ct-punc` | -- | -- | -- | Standalone only. Takes text input. |
| `fsmn-vad` | -- | -- | -- | Standalone only. |

### Pipeline example

```python
# Long Chinese audio: paraformer-zh + VAD + punctuation
model = asr.load_model("paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
result = model(audio="meeting.wav")

# Long audio: SenseVoiceSmall + VAD (no punc)
model = asr.load_model("SenseVoiceSmall", vad_model="fsmn-vad")
result = model(audio="long_audio.wav")
```

## Input Methods

All audio models accept three input types:

```python
model = asr.load_model("SenseVoiceSmall")

# 1. File path
result = model(audio="audio.wav")

# 2. Raw bytes
audio_bytes = Path("audio.wav").read_bytes()
result = model(audio_bytes=audio_bytes)

# 3. Text (for punctuation models only)
punc = asr.load_model("ct-punc")
result = punc(text="你好世界今天天气真好")
```

## All Available Models

| Name | Type | Params | Description |
|------|------|--------|-------------|
| `SenseVoiceSmall` | asr | 234M | Multi-task ASR, zh/en/ja/ko/yue, emotion + event tags |
| `Fun-ASR-Nano` | asr | 800M | End-to-end ASR, built-in punctuation + timestamps |
| `Fun-ASR-MLT-Nano` | asr | 800M | Multilingual ASR, 31 languages |
| `paraformer` | asr | 220M | Offline, zh + en, max 20s |
| `paraformer-zh` | asr | 220M | Offline, zh + en, arbitrary length (with SeACo) |
| `paraformer-en` | asr | 220M | Offline, English |
| `paraformer-en-spk` | asr | 220M | English + built-in speaker diarization |
| `paraformer-zh-streaming` | asr | 220M | Streaming, zh + en |
| `Whisper-large-v2` | asr | 1550M | OpenAI Whisper large-v2, multilingual |
| `Whisper-large-v3` | asr | 1550M | OpenAI Whisper large-v3, multilingual |
| `Whisper-large-v3-turbo` | asr | 809M | OpenAI Whisper large-v3 turbo |
| `fsmn-vad` | vad | 0.4M | Voice activity detection, 16kHz |
| `ct-punc` | punc | 1.1G | Punctuation restoration, zh + en |
| `ct-punc-c` | punc | 291M | Punctuation restoration (compact), zh + en |
| `cam++` | spk | 7.2M | Speaker embedding, 192-dim |
| `fa-zh` | fa | 37.8M | Forced alignment / timestamp prediction, zh |
| `emotion2vec_plus_large` | emotion | 300M | Emotion recognition, 9 classes |
| `emotion2vec_plus_base` | emotion | - | Emotion recognition (base) |
| `emotion2vec_plus_seed` | emotion | - | Emotion recognition (seed) |

Model names are automatically resolved to the correct hub (ModelScope in China, HuggingFace internationally).

## API Reference

### `FunASR(runtime_dir, port, host)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `runtime_dir` | `"./funasr_runtime"` | Directory for the server environment |
| `port` | `0` (auto) | Server port |
| `host` | `"127.0.0.1"` | Bind host |

### FunASR Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `ensure_installed()` | `bool` | Install runtime (one-time). Returns True if already installed. |
| `start(timeout=60)` | `int` | Start server, returns port number. |
| `stop()` | - | Stop the server. |
| `load_model(model, ...)` | `Model` | Load a model, returns a `Model` handle. |
| `health()` | `dict` | Check server status. |
| `list_models()` | `dict` | List loaded models. |
| `get_progress(name)` | `dict` | Get inference progress `{"current", "total"}`. |
| `execute(code)` | `dict` | Execute Python code on the server. |

### `load_model()` Parameters

```python
model = asr.load_model(
    model,                  # Required: model name ("SenseVoiceSmall", "fsmn-vad", etc.)
    vad_model=None,         # VAD model for pipeline
    punc_model=None,        # Punctuation model for pipeline
    spk_model=None,         # Speaker model for pipeline
    device=None,            # "cuda" / "cpu" / None (auto)
    hub=None,               # "ms" / "hf" / None (auto)
    quantize=None,          # Enable quantization
    fp16=None,              # Enable half-precision
    batch_size=None,        # Batch size
    disable_update=None,    # Skip model update checks
)
```

### Model Methods

```python
model = asr.load_model("SenseVoiceSmall")

# Inference
result = model.infer(audio="file.wav")
result = model.infer(audio_bytes=raw_bytes)
result = model.infer(text="input text")

# Shorthand
result = model(audio="file.wav")

# Alias for ASR
result = model.transcribe(audio="file.wav")

# Progress query
progress = model.get_progress()  # {"current": 3, "total": 10}

# Unload from memory
model.unload()
```

**Inference parameters** (passed to `infer()` or `__call__()`):

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | `str` | Path to audio file |
| `audio_bytes` | `bytes` | Raw audio bytes |
| `text` | `str` | Text input (for punctuation models) |
| `language` | `str` | Language hint (`"zh"`, `"en"`, `"ja"`, etc.) |
| `use_itn` | `bool` | Enable inverse text normalization |
| `batch_size` | `int` | Inference batch size |
| `hotword` | `str` | Hotword string for biased recognition |
| `merge_vad` | `bool` | Merge short VAD segments |
| `merge_length_s` | `float` | Max merge length in seconds (default: 15) |
| `progress_callback` | `callable` | Progress callback `(current, total) -> None` |

### Inference Progress

You can track inference progress using `progress_callback`:

```python
model = asr.load_model("SenseVoiceSmall", vad_model="fsmn-vad")

def on_progress(current, total):
    if total > 0:
        print(f"\rProgress: {current}/{total} ({current/total*100:.0f}%)", end="")

result = model.infer(audio="long_meeting.wav", progress_callback=on_progress)
```

When `progress_callback` is provided, inference runs in a background thread while the client polls the server every 0.5s for progress updates. The callback receives `(current, total)` where `current` is the number of completed batches and `total` is the total number of batches.

You can also query progress manually (e.g. from another thread):

```python
progress = model.get_progress()  # {"current": 3, "total": 10}
```

When no inference is running, returns `{"current": 0, "total": 0}`.

> **Note:** Progress granularity depends on the number of VAD segments. Short audio with few segments may only show 0/0 → 1/1. Longer audio (e.g. meetings) with many VAD segments will produce finer-grained progress updates.

## Architecture

```
Your Application
    |
    |  HTTP (localhost)
    |  JSON-RPC 2.0
    v
FunASR Server (background process)
    |
    |-- Models loaded in memory
    |-- Isolated Python environment (uv)
    +-- Auto GPU/CPU detection
```

The server runs in a completely isolated Python environment managed by `uv`. Your application communicates with it over HTTP using JSON-RPC 2.0 protocol.

## Requirements

- Python >= 3.10 (for the client SDK only)
- Internet connection (for first-time setup)
- `curl` (Linux/macOS) or PowerShell (Windows) — for auto-installing uv

## License

MIT
