"""Integration tests — real environment installation and model inference.

These tests actually:
1. Install the full FunASR runtime (uv + torch + funasr)
2. Start the server process
3. Load real models and run inference
4. Verify results

Requires: internet connection, ~2GB disk space, ~5 minutes.

Run with:
    pytest tests/test_integration.py -v -s

Skip in normal CI (these are marked with @pytest.mark.integration).
"""

import math
import struct
import tempfile
import wave
from pathlib import Path

import pytest

from funasr_server import FunASR


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def _generate_test_wav(path: str, duration_sec: float = 2.0, sample_rate: int = 16000):
    """Generate a simple sine wave WAV file using only the standard library.

    Creates a 440Hz tone — enough for VAD to detect speech-like activity.
    """
    frequency = 440.0
    num_samples = int(duration_sec * sample_rate)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)

        for i in range(num_samples):
            sample = int(32767 * 0.5 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))


@pytest.fixture(scope="module")
def runtime_dir():
    """Temporary directory for the FunASR runtime."""
    with tempfile.TemporaryDirectory(prefix="funasr_test_") as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def test_audio():
    """Generate a temporary test audio file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        _generate_test_wav(f.name, duration_sec=2.0)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def client(runtime_dir):
    """Create and start a FunASR client with a real runtime environment."""
    asr = FunASR(runtime_dir=runtime_dir)
    asr.ensure_installed()
    asr.start(timeout=120)
    yield asr
    asr.stop()


class TestLifecycle:
    """Test installation and server lifecycle."""

    def test_ensure_installed(self, runtime_dir):
        """Runtime environment can be installed."""
        asr = FunASR(runtime_dir=runtime_dir)
        # Should already be installed by the client fixture, or install now
        result = asr.ensure_installed()
        # True = already installed, False = just installed. Both are ok.
        assert isinstance(result, bool)

        # Verify files exist
        rt = Path(runtime_dir)
        assert (rt / ".venv").exists()
        assert (rt / "pyproject.toml").exists()
        assert (rt / "server.py").exists()

    def test_health(self, client):
        """Server responds to health check."""
        result = client.health()
        assert result["status"] == "ok"
        assert isinstance(result["loaded_models"], list)
        assert "cuda_available" in result

    def test_is_running(self, client):
        """Server process is alive."""
        assert client.is_running() is True


class TestVADModel:
    """Test with FSMN-VAD (smallest model, ~36MB)."""

    def test_load_vad_model(self, client):
        """Load the FSMN-VAD model."""
        result = client.load_model(
            model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            name="vad",
        )
        assert result["status"] == "loaded"
        assert result["name"] == "vad"

    def test_infer_vad(self, client, test_audio):
        """VAD inference returns speech segments."""
        result = client.infer(input=test_audio, name="vad")

        assert isinstance(result, list)
        assert len(result) > 0

        # VAD returns [{"key": ..., "value": [[start_ms, end_ms], ...]}]
        first = result[0]
        assert "key" in first
        assert "value" in first

        segments = first["value"]
        assert isinstance(segments, list)
        # Should detect at least one segment in our 2-second tone
        assert len(segments) > 0

        # Each segment is [start_ms, end_ms]
        for seg in segments:
            assert len(seg) == 2
            assert seg[0] >= 0
            assert seg[1] > seg[0]

    def test_infer_vad_with_bytes(self, client, test_audio):
        """VAD inference works with audio bytes input."""
        audio_bytes = Path(test_audio).read_bytes()
        result = client.infer(input_bytes=audio_bytes, name="vad")

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

        # Verify it's gone
        models = client.list_models()
        assert "vad" not in models["models"]


class TestASRModel:
    """Test with SenseVoiceSmall (~450MB)."""

    def test_load_asr_model(self, client):
        """Load SenseVoiceSmall ASR model."""
        result = client.load_model(
            model="iic/SenseVoiceSmall",
            name="asr",
        )
        assert result["status"] == "loaded"
        assert result["name"] == "asr"

    def test_infer_asr(self, client, test_audio):
        """ASR inference returns transcription text."""
        result = client.infer(input=test_audio, name="asr")

        assert isinstance(result, list)
        assert len(result) > 0

        first = result[0]
        assert "key" in first
        assert "text" in first
        assert isinstance(first["text"], str)
        assert len(first["text"]) > 0

    def test_infer_asr_with_bytes(self, client, test_audio):
        """ASR inference works with audio bytes input."""
        audio_bytes = Path(test_audio).read_bytes()
        result = client.infer(input_bytes=audio_bytes, name="asr")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]

    def test_transcribe_alias(self, client, test_audio):
        """transcribe() works as alias for infer()."""
        result = client.transcribe(audio=test_audio, name="asr")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]

    def test_unload_asr(self, client):
        """Unload ASR model."""
        result = client.unload_model(name="asr")
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

    def test_execute_with_output(self, client):
        """Capture stdout from executed code."""
        result = client.execute("print('hello from server')")
        assert "hello from server" in result["output"]

    def test_execute_error(self, client):
        """Errors in executed code are reported properly."""
        result = client.execute("raise ValueError('test error')")
        assert result.get("error") is not None
        assert "test error" in result["error"]
