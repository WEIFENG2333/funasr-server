"""Integration tests — real environment installation and model inference.

These tests actually:
1. Install the full FunASR runtime (uv + torch + funasr)
2. Start the server process
3. Load real models and run inference
4. Verify results with real audio

All network detection and model name resolution is handled by the
package itself (mirror.py + models.py), not by test-side logic.

Requires: internet connection, ~2GB disk space, ~10 minutes.

Run with:
    pytest tests/test_integration.py -v -s
"""

import time
import tempfile
from pathlib import Path

import pytest

from funasr_server import FunASR, detect_region, get_hub


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

# Test audio file (real Chinese speech, ~5.5s, 16kHz mono)
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_TEST_AUDIO = str(_FIXTURES_DIR / "asr_example.wav")


def _print_metric(name: str, value, unit: str = ""):
    """Print a test metric for debugging visibility."""
    label = f"  [METRIC] {name}: {value}"
    if unit:
        label += f" {unit}"
    print(label)


@pytest.fixture(scope="module")
def runtime_dir():
    """Temporary directory for the FunASR runtime."""
    with tempfile.TemporaryDirectory(prefix="funasr_test_") as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def client(runtime_dir):
    """Create and start a FunASR client with a real runtime environment."""
    asr = FunASR(runtime_dir=runtime_dir)

    t0 = time.time()
    asr.ensure_installed()
    install_time = time.time() - t0
    print(f"\n  [METRIC] install_time: {install_time:.1f} s")

    t0 = time.time()
    asr.start(timeout=120)
    start_time = time.time() - t0
    print(f"  [METRIC] server_start_time: {start_time:.1f} s")

    yield asr
    asr.stop()


class TestLifecycle:
    """Test installation and server lifecycle."""

    def test_ensure_installed(self, runtime_dir):
        """Runtime environment can be installed."""
        asr = FunASR(runtime_dir=runtime_dir)
        result = asr.ensure_installed()
        assert isinstance(result, bool)

        rt = Path(runtime_dir)
        assert (rt / ".venv").exists()
        assert (rt / "pyproject.toml").exists()
        assert (rt / "server.py").exists()

    def test_health(self, client):
        """Server responds to health check."""
        t0 = time.time()
        result = client.health()
        _print_metric("health_latency", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert result["status"] == "ok"
        assert isinstance(result["loaded_models"], list)
        assert "cuda_available" in result
        _print_metric("cuda_available", result["cuda_available"])
        _print_metric("device", result["device"])

    def test_is_running(self, client):
        """Server process is alive."""
        assert client.is_running() is True

    def test_region_and_hub_detected(self):
        """Package auto-detects region and hub correctly."""
        region = detect_region()
        hub = get_hub()
        _print_metric("detected_region", region)
        _print_metric("detected_hub", hub)
        assert region in ("cn", "intl")
        assert hub in ("ms", "hf")


class TestVADModel:
    """Test with FSMN-VAD (smallest model, ~36MB)."""

    def test_load_vad_model(self, client):
        """Load the FSMN-VAD model — uses package's auto hub resolution."""
        t0 = time.time()
        result = client.load_model(model="fsmn-vad", name="vad")
        load_time = time.time() - t0
        _print_metric("vad_load_time", f"{load_time:.1f}", "s")

        assert result["status"] == "loaded"
        assert result["name"] == "vad"

    def test_infer_vad(self, client):
        """VAD inference returns speech segments from real audio."""
        t0 = time.time()
        result = client.infer(input=_TEST_AUDIO, name="vad")
        infer_time = time.time() - t0
        _print_metric("vad_infer_time", f"{infer_time * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0

        first = result[0]
        assert "key" in first
        assert "value" in first

        segments = first["value"]
        _print_metric("vad_segments_count", len(segments))
        assert isinstance(segments, list)
        assert len(segments) > 0

        for seg in segments:
            assert len(seg) == 2
            assert seg[0] >= 0
            assert seg[1] > seg[0]
            _print_metric("vad_segment", f"[{seg[0]}ms, {seg[1]}ms]")

    def test_infer_vad_with_bytes(self, client):
        """VAD inference works with audio bytes input."""
        audio_bytes = Path(_TEST_AUDIO).read_bytes()
        _print_metric("audio_bytes_size", f"{len(audio_bytes) / 1024:.1f}", "KB")

        t0 = time.time()
        result = client.infer(input_bytes=audio_bytes, name="vad")
        _print_metric("vad_infer_bytes_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "value" in result[0]

    def test_list_models_shows_vad(self, client):
        """Loaded VAD model appears in model list."""
        result = client.list_models()
        assert "vad" in result["models"]

    def test_unload_vad(self, client):
        """Unload VAD model."""
        result = client.unload_model(name="vad")
        assert result["status"] == "unloaded"

        models = client.list_models()
        assert "vad" not in models["models"]


class TestASRModel:
    """Test with SenseVoiceSmall (~450MB)."""

    def test_load_asr_model(self, client):
        """Load SenseVoiceSmall — uses package's auto model name resolution."""
        t0 = time.time()
        result = client.load_model(model="SenseVoiceSmall", name="asr")
        load_time = time.time() - t0
        _print_metric("asr_load_time", f"{load_time:.1f}", "s")

        assert result["status"] == "loaded"
        assert result["name"] == "asr"

    def test_infer_asr(self, client):
        """ASR inference returns transcription text from real audio."""
        t0 = time.time()
        result = client.infer(input=_TEST_AUDIO, name="asr")
        infer_time = time.time() - t0
        _print_metric("asr_infer_time", f"{infer_time * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0

        first = result[0]
        assert "key" in first
        assert "text" in first
        assert isinstance(first["text"], str)
        assert len(first["text"]) > 0
        _print_metric("asr_text", repr(first["text"]))

    def test_infer_asr_with_bytes(self, client):
        """ASR inference works with audio bytes input."""
        audio_bytes = Path(_TEST_AUDIO).read_bytes()

        t0 = time.time()
        result = client.infer(input_bytes=audio_bytes, name="asr")
        _print_metric("asr_infer_bytes_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]
        _print_metric("asr_text_bytes", repr(result[0]["text"]))

    def test_transcribe_alias(self, client):
        """transcribe() works as alias for infer()."""
        result = client.transcribe(audio=_TEST_AUDIO, name="asr")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]

    def test_unload_asr(self, client):
        """Unload ASR model."""
        result = client.unload_model(name="asr")
        assert result["status"] == "unloaded"


class TestSpeakerModel:
    """Test with CAM++ speaker verification model."""

    def test_load_speaker_model(self, client):
        """Load cam++ speaker embedding model."""
        t0 = time.time()
        result = client.load_model(model="cam++", name="spk")
        load_time = time.time() - t0
        _print_metric("spk_load_time", f"{load_time:.1f}", "s")

        assert result["status"] == "loaded"
        assert result["name"] == "spk"

    def test_infer_speaker_embedding(self, client):
        """Speaker model extracts embeddings from real audio."""
        t0 = time.time()
        result = client.infer(input=_TEST_AUDIO, name="spk")
        infer_time = time.time() - t0
        _print_metric("spk_infer_time", f"{infer_time * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        _print_metric("spk_result_keys", list(result[0].keys()))

    def test_unload_speaker(self, client):
        """Unload speaker model."""
        result = client.unload_model(name="spk")
        assert result["status"] == "unloaded"


class TestASRWithVADPipeline:
    """Test ASR with VAD model for pipeline composition."""

    def test_load_asr_with_vad(self, client):
        """Load ASR model with VAD pipeline — model names auto-resolved."""
        t0 = time.time()
        result = client.load_model(
            model="SenseVoiceSmall",
            vad_model="fsmn-vad",
            name="asr_vad",
        )
        load_time = time.time() - t0
        _print_metric("asr_vad_load_time", f"{load_time:.1f}", "s")

        assert result["status"] == "loaded"
        assert result["name"] == "asr_vad"

    def test_infer_asr_with_vad(self, client):
        """ASR+VAD pipeline inference on real audio."""
        t0 = time.time()
        result = client.infer(input=_TEST_AUDIO, name="asr_vad")
        infer_time = time.time() - t0
        _print_metric("asr_vad_infer_time", f"{infer_time * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0

        first = result[0]
        assert "text" in first
        assert len(first["text"]) > 0
        _print_metric("asr_vad_text", repr(first["text"]))

    def test_unload_asr_vad(self, client):
        """Unload ASR+VAD pipeline model."""
        result = client.unload_model(name="asr_vad")
        assert result["status"] == "unloaded"


class TestPunctuationModel:
    """Test with CT-Punc punctuation restoration model."""

    def test_load_punc_model(self, client):
        """Load ct-punc punctuation model."""
        t0 = time.time()
        result = client.load_model(model="ct-punc", name="punc")
        load_time = time.time() - t0
        _print_metric("punc_load_time", f"{load_time:.1f}", "s")

        assert result["status"] == "loaded"
        assert result["name"] == "punc"

    def test_infer_punctuation(self, client):
        """Punctuation model adds punctuation to text input."""
        test_text = "你好世界今天天气真好我们一起出去玩吧"

        t0 = time.time()
        result = client.infer(input=test_text, name="punc")
        infer_time = time.time() - t0
        _print_metric("punc_infer_time", f"{infer_time * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0

        first = result[0]
        assert "text" in first
        punctuated = first["text"]
        _print_metric("punc_input", repr(test_text))
        _print_metric("punc_output", repr(punctuated))
        assert len(punctuated) > 0

    def test_unload_punc(self, client):
        """Unload punctuation model."""
        result = client.unload_model(name="punc")
        assert result["status"] == "unloaded"


class TestExecute:
    """Test arbitrary code execution."""

    def test_execute_simple(self, client):
        """Execute simple Python code."""
        result = client.execute("result = 1 + 1")
        assert result["return_value"] == 2
        assert result.get("error") is None

    def test_execute_import(self, client):
        """Import and use libraries in the server environment."""
        result = client.execute(
            "import torch; result = torch.cuda.is_available()"
        )
        assert result.get("error") is None
        assert isinstance(result["return_value"], bool)
        _print_metric("torch_cuda_available", result["return_value"])

    def test_execute_with_output(self, client):
        """Capture stdout from executed code."""
        result = client.execute("print('hello from server')")
        assert "hello from server" in result["output"]

    def test_execute_error(self, client):
        """Errors in executed code are reported properly."""
        result = client.execute("raise ValueError('test error')")
        assert result.get("error") is not None
        assert "test error" in result["error"]
