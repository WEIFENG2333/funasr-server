"""Integration tests — real environment installation and model inference.

These tests actually:
1. Install the full FunASR runtime (uv + torch + funasr)
2. Start the server process
3. Load real models and run inference
4. Verify results with real audio

Requires: internet connection, ~2GB disk space, ~10 minutes.

Run with:
    pytest tests/test_integration.py -v -s
"""

import time
import tempfile
from pathlib import Path

import pytest

from funasr_server import FunASR, detect_region, get_hub


pytestmark = pytest.mark.integration

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_TEST_AUDIO = str(_FIXTURES_DIR / "asr_example.wav")


def _print_metric(name: str, value, unit: str = ""):
    label = f"  [METRIC] {name}: {value}"
    if unit:
        label += f" {unit}"
    print(label)


@pytest.fixture(scope="module")
def runtime_dir():
    with tempfile.TemporaryDirectory(prefix="funasr_test_") as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def client(runtime_dir):
    asr = FunASR(runtime_dir=runtime_dir)

    t0 = time.time()
    asr.ensure_installed()
    _print_metric("install_time", f"{time.time() - t0:.1f}", "s")

    t0 = time.time()
    asr.start(timeout=120)
    _print_metric("server_start_time", f"{time.time() - t0:.1f}", "s")

    yield asr
    asr.stop()


class TestLifecycle:

    def test_ensure_installed(self, runtime_dir):
        asr = FunASR(runtime_dir=runtime_dir)
        result = asr.ensure_installed()
        assert isinstance(result, bool)

        rt = Path(runtime_dir)
        assert (rt / ".venv").exists()
        assert (rt / "pyproject.toml").exists()
        assert (rt / "server.py").exists()

    def test_health(self, client):
        t0 = time.time()
        result = client.health()
        _print_metric("health_latency", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert result["status"] == "ok"
        assert isinstance(result["loaded_models"], list)
        _print_metric("cuda_available", result["cuda_available"])
        _print_metric("device", result["device"])

    def test_is_running(self, client):
        assert client.is_running() is True

    def test_region_and_hub_detected(self):
        region = detect_region()
        hub = get_hub()
        _print_metric("detected_region", region)
        _print_metric("detected_hub", hub)
        assert region in ("cn", "intl")
        assert hub in ("ms", "hf")


class TestVADModel:

    def test_load_and_infer(self, client):
        t0 = time.time()
        vad = client.load_model("fsmn-vad")
        _print_metric("vad_load_time", f"{time.time() - t0:.1f}", "s")

        # Infer with file path
        t0 = time.time()
        result = vad.infer(audio=_TEST_AUDIO)
        _print_metric("vad_infer_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        segments = result[0]["value"]
        _print_metric("vad_segments_count", len(segments))
        assert len(segments) > 0
        for seg in segments:
            assert len(seg) == 2
            assert seg[1] > seg[0]

        # Infer with bytes
        audio_bytes = Path(_TEST_AUDIO).read_bytes()
        _print_metric("audio_bytes_size", f"{len(audio_bytes) / 1024:.1f}", "KB")
        result_bytes = vad(audio_bytes=audio_bytes)
        assert len(result_bytes) > 0
        assert "value" in result_bytes[0]

        # List models
        models = client.list_models()
        assert vad.name in models["models"]

        # Unload
        result = vad.unload()
        assert result["status"] == "unloaded"
        assert vad.name not in client.list_models()["models"]


class TestASRModel:

    def test_load_and_infer(self, client):
        t0 = time.time()
        asr = client.load_model("SenseVoiceSmall")
        _print_metric("asr_load_time", f"{time.time() - t0:.1f}", "s")

        # Infer with file path
        t0 = time.time()
        result = asr.infer(audio=_TEST_AUDIO)
        _print_metric("asr_infer_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]
        assert len(result[0]["text"]) > 0
        _print_metric("asr_text", repr(result[0]["text"]))

        # Infer with bytes
        audio_bytes = Path(_TEST_AUDIO).read_bytes()
        result_bytes = asr(audio_bytes=audio_bytes)
        assert "text" in result_bytes[0]
        _print_metric("asr_text_bytes", repr(result_bytes[0]["text"]))

        # transcribe alias
        result_t = asr.transcribe(audio=_TEST_AUDIO)
        assert "text" in result_t[0]

        asr.unload()


class TestNanoModel:

    def test_load_and_infer(self, client):
        t0 = time.time()
        nano = client.load_model("Fun-ASR-Nano")
        _print_metric("nano_load_time", f"{time.time() - t0:.1f}", "s")

        t0 = time.time()
        result = nano.infer(audio=_TEST_AUDIO)
        _print_metric("nano_infer_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]
        assert len(result[0]["text"]) > 0
        _print_metric("nano_text", repr(result[0]["text"]))

        nano.unload()


class TestSpeakerModel:

    def test_load_and_infer(self, client):
        t0 = time.time()
        spk = client.load_model("cam++")
        _print_metric("spk_load_time", f"{time.time() - t0:.1f}", "s")

        t0 = time.time()
        result = spk.infer(audio=_TEST_AUDIO)
        _print_metric("spk_infer_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        _print_metric("spk_result_keys", list(result[0].keys()))

        spk.unload()


class TestASRWithVADPipeline:

    def test_load_and_infer(self, client):
        t0 = time.time()
        model = client.load_model("SenseVoiceSmall", vad_model="fsmn-vad")
        _print_metric("asr_vad_load_time", f"{time.time() - t0:.1f}", "s")

        t0 = time.time()
        result = model.infer(audio=_TEST_AUDIO)
        _print_metric("asr_vad_infer_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        assert "text" in result[0]
        assert len(result[0]["text"]) > 0
        _print_metric("asr_vad_text", repr(result[0]["text"]))

        model.unload()


class TestPunctuationModel:

    def test_load_and_infer(self, client):
        t0 = time.time()
        punc = client.load_model("ct-punc")
        _print_metric("punc_load_time", f"{time.time() - t0:.1f}", "s")

        test_text = "你好世界今天天气真好我们一起出去玩吧"
        t0 = time.time()
        result = punc.infer(text=test_text)
        _print_metric("punc_infer_time", f"{(time.time() - t0) * 1000:.0f}", "ms")

        assert isinstance(result, list)
        assert len(result) > 0
        punctuated = result[0]["text"]
        _print_metric("punc_input", repr(test_text))
        _print_metric("punc_output", repr(punctuated))
        assert len(punctuated) > 0

        punc.unload()


class TestExecute:

    def test_execute_simple(self, client):
        result = client.execute("result = 1 + 1")
        assert result["return_value"] == 2
        assert result.get("error") is None

    def test_execute_import(self, client):
        result = client.execute("import torch; result = torch.cuda.is_available()")
        assert result.get("error") is None
        assert isinstance(result["return_value"], bool)
        _print_metric("torch_cuda_available", result["return_value"])

    def test_execute_with_output(self, client):
        result = client.execute("print('hello from server')")
        assert "hello from server" in result["output"]

    def test_execute_error(self, client):
        result = client.execute("raise ValueError('test error')")
        assert result.get("error") is not None
        assert "test error" in result["error"]
