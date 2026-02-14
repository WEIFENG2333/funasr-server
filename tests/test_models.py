"""Tests for the model name registry."""

import pytest

from funasr_server.models import (
    MODEL_REGISTRY,
    resolve_model_id,
    list_available_models,
    _ALIAS_MAP,
    _build_alias_map,
)
import funasr_server.models as models_module


@pytest.fixture(autouse=True)
def reset_alias_map():
    """Reset alias map between tests to ensure clean state."""
    _ALIAS_MAP.clear()
    yield
    _ALIAS_MAP.clear()


# ------------------------------------------------------------------
# resolve_model_id — SenseVoice
# ------------------------------------------------------------------

def test_resolve_sensevoicesmall_ms():
    assert resolve_model_id("SenseVoiceSmall", hub="ms") == "iic/SenseVoiceSmall"


def test_resolve_sensevoicesmall_hf():
    assert resolve_model_id("SenseVoiceSmall", hub="hf") == "FunAudioLLM/SenseVoiceSmall"


# ------------------------------------------------------------------
# resolve_model_id — Fun-ASR-Nano
# ------------------------------------------------------------------

def test_resolve_fun_asr_nano_ms():
    assert resolve_model_id("Fun-ASR-Nano", hub="ms") == "FunAudioLLM/Fun-ASR-Nano-2512"


def test_resolve_fun_asr_nano_hf():
    assert resolve_model_id("Fun-ASR-Nano", hub="hf") == "FunAudioLLM/Fun-ASR-Nano-2512"


def test_resolve_fun_asr_mlt_nano():
    assert resolve_model_id("Fun-ASR-MLT-Nano", hub="hf") == "FunAudioLLM/Fun-ASR-MLT-Nano-2512"


# ------------------------------------------------------------------
# resolve_model_id — Paraformer family
# ------------------------------------------------------------------

def test_resolve_paraformer():
    assert resolve_model_id("paraformer", hub="ms") == "paraformer"
    assert resolve_model_id("paraformer", hub="hf") == "paraformer"


def test_resolve_paraformer_zh():
    assert resolve_model_id("paraformer-zh", hub="ms") == "paraformer-zh"


def test_resolve_paraformer_en():
    assert resolve_model_id("paraformer-en", hub="hf") == "paraformer-en"


def test_resolve_paraformer_en_spk():
    assert resolve_model_id("paraformer-en-spk", hub="ms") == "paraformer-en-spk"


def test_resolve_paraformer_streaming():
    assert resolve_model_id("paraformer-zh-streaming", hub="hf") == "paraformer-zh-streaming"


# ------------------------------------------------------------------
# resolve_model_id — VAD / Punc / Speaker
# ------------------------------------------------------------------

def test_resolve_fsmn_vad():
    assert resolve_model_id("fsmn-vad", hub="ms") == "fsmn-vad"
    assert resolve_model_id("fsmn-vad", hub="hf") == "fsmn-vad"


def test_resolve_ct_punc():
    assert resolve_model_id("ct-punc", hub="ms") == "ct-punc"
    assert resolve_model_id("ct-punc", hub="hf") == "ct-punc"


def test_resolve_ct_punc_c():
    assert resolve_model_id("ct-punc-c", hub="ms") == "ct-punc-c"


def test_resolve_cam_plus():
    assert resolve_model_id("cam++", hub="ms") == "cam++"
    assert resolve_model_id("cam++", hub="hf") == "cam++"


# ------------------------------------------------------------------
# resolve_model_id — Whisper / Qwen
# ------------------------------------------------------------------

def test_resolve_whisper_v2():
    assert resolve_model_id("Whisper-large-v2", hub="ms") == "Whisper-large-v2"


def test_resolve_whisper_v3():
    assert resolve_model_id("Whisper-large-v3", hub="hf") == "Whisper-large-v3"


def test_resolve_whisper_turbo():
    assert resolve_model_id("Whisper-large-v3-turbo", hub="ms") == "Whisper-large-v3-turbo"


def test_resolve_qwen_audio():
    assert resolve_model_id("Qwen-Audio", hub="ms") == "Qwen-Audio"


# ------------------------------------------------------------------
# resolve_model_id — Emotion / FA
# ------------------------------------------------------------------

def test_resolve_emotion2vec():
    assert resolve_model_id("emotion2vec_plus_large", hub="hf") == "emotion2vec_plus_large"
    assert resolve_model_id("emotion2vec_plus_base", hub="ms") == "emotion2vec_plus_base"
    assert resolve_model_id("emotion2vec_plus_seed", hub="hf") == "emotion2vec_plus_seed"


def test_resolve_fa_zh():
    assert resolve_model_id("fa-zh", hub="ms") == "fa-zh"


# ------------------------------------------------------------------
# resolve_model_id — case insensitive / reverse lookup
# ------------------------------------------------------------------

def test_resolve_case_insensitive():
    assert resolve_model_id("sensevoicesmall", hub="hf") == "FunAudioLLM/SenseVoiceSmall"
    assert resolve_model_id("SENSEVOICESMALL", hub="ms") == "iic/SenseVoiceSmall"
    assert resolve_model_id("fun-asr-nano", hub="hf") == "FunAudioLLM/Fun-ASR-Nano-2512"


def test_resolve_by_hub_id():
    """Can resolve by hub-specific ID (reverse lookup)."""
    assert resolve_model_id("iic/SenseVoiceSmall", hub="hf") == "FunAudioLLM/SenseVoiceSmall"
    assert resolve_model_id("FunAudioLLM/SenseVoiceSmall", hub="ms") == "iic/SenseVoiceSmall"


def test_resolve_nano_by_full_id():
    assert resolve_model_id("FunAudioLLM/Fun-ASR-Nano-2512", hub="ms") == "FunAudioLLM/Fun-ASR-Nano-2512"


def test_resolve_unknown_model():
    assert resolve_model_id("some/unknown-model", hub="hf") == "some/unknown-model"
    assert resolve_model_id("my-custom-model", hub="ms") == "my-custom-model"


def test_resolve_default_hub_is_ms():
    assert resolve_model_id("SenseVoiceSmall") == "iic/SenseVoiceSmall"


# ------------------------------------------------------------------
# resolve_model_id — all registry models
# ------------------------------------------------------------------

def test_resolve_all_registry_models():
    """Every model in the registry resolves successfully on both hubs."""
    for name, entry in MODEL_REGISTRY.items():
        ms_id = resolve_model_id(name, hub="ms")
        hf_id = resolve_model_id(name, hub="hf")
        assert ms_id == entry["ms"], f"{name}: ms expected {entry['ms']}, got {ms_id}"
        assert hf_id == entry["hf"], f"{name}: hf expected {entry['hf']}, got {hf_id}"


# ------------------------------------------------------------------
# MODEL_REGISTRY structure
# ------------------------------------------------------------------

def test_registry_has_required_keys():
    """Every registry entry has ms, hf, type, and desc keys."""
    for name, entry in MODEL_REGISTRY.items():
        assert "ms" in entry, f"{name} missing 'ms' key"
        assert "hf" in entry, f"{name} missing 'hf' key"
        assert "type" in entry, f"{name} missing 'type' key"
        assert "desc" in entry, f"{name} missing 'desc' key"


def test_registry_model_types():
    """Registry contains models of expected types."""
    types = {entry["type"] for entry in MODEL_REGISTRY.values()}
    assert "asr" in types
    assert "vad" in types
    assert "punc" in types
    assert "spk" in types
    assert "emotion" in types
    assert "fa" in types


def test_registry_has_common_models():
    """Registry includes the most commonly used models."""
    assert "SenseVoiceSmall" in MODEL_REGISTRY
    assert "Fun-ASR-Nano" in MODEL_REGISTRY
    assert "Fun-ASR-MLT-Nano" in MODEL_REGISTRY
    assert "paraformer-zh" in MODEL_REGISTRY
    assert "fsmn-vad" in MODEL_REGISTRY
    assert "ct-punc" in MODEL_REGISTRY
    assert "cam++" in MODEL_REGISTRY
    assert "Whisper-large-v3" in MODEL_REGISTRY
    assert "emotion2vec_plus_large" in MODEL_REGISTRY


def test_registry_model_count():
    """Registry has the expected number of models."""
    assert len(MODEL_REGISTRY) >= 20


# ------------------------------------------------------------------
# list_available_models
# ------------------------------------------------------------------

def test_list_available_models():
    models = list_available_models()
    assert isinstance(models, dict)
    assert len(models) == len(MODEL_REGISTRY)
    assert "SenseVoiceSmall" in models
    assert models["SenseVoiceSmall"]["type"] == "asr"
    assert "desc" in models["SenseVoiceSmall"]


def test_list_available_models_returns_copy():
    """list_available_models returns a copy, not the original."""
    models = list_available_models()
    models["SenseVoiceSmall"]["type"] = "modified"
    assert MODEL_REGISTRY["SenseVoiceSmall"]["type"] == "asr"
