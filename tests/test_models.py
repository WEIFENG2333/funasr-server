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
# resolve_model_id
# ------------------------------------------------------------------

def test_resolve_sensevoicesmall_ms():
    """SenseVoiceSmall resolves to iic/SenseVoiceSmall on ModelScope."""
    result = resolve_model_id("SenseVoiceSmall", hub="ms")
    assert result == "iic/SenseVoiceSmall"


def test_resolve_sensevoicesmall_hf():
    """SenseVoiceSmall resolves to FunAudioLLM/SenseVoiceSmall on HuggingFace."""
    result = resolve_model_id("SenseVoiceSmall", hub="hf")
    assert result == "FunAudioLLM/SenseVoiceSmall"


def test_resolve_fsmn_vad_ms():
    result = resolve_model_id("fsmn-vad", hub="ms")
    assert result == "fsmn-vad"


def test_resolve_fsmn_vad_hf():
    result = resolve_model_id("fsmn-vad", hub="hf")
    assert result == "fsmn-vad"


def test_resolve_ct_punc():
    assert resolve_model_id("ct-punc", hub="ms") == "ct-punc"
    assert resolve_model_id("ct-punc", hub="hf") == "ct-punc"


def test_resolve_cam_plus():
    assert resolve_model_id("cam++", hub="ms") == "cam++"
    assert resolve_model_id("cam++", hub="hf") == "cam++"


def test_resolve_case_insensitive():
    """Model name lookup is case-insensitive."""
    assert resolve_model_id("sensevoicesmall", hub="hf") == "FunAudioLLM/SenseVoiceSmall"
    assert resolve_model_id("SENSEVOICESMALL", hub="ms") == "iic/SenseVoiceSmall"


def test_resolve_by_hub_id():
    """Can also resolve by hub-specific ID (reverse lookup)."""
    assert resolve_model_id("iic/SenseVoiceSmall", hub="hf") == "FunAudioLLM/SenseVoiceSmall"
    assert resolve_model_id("FunAudioLLM/SenseVoiceSmall", hub="ms") == "iic/SenseVoiceSmall"


def test_resolve_unknown_model():
    """Unknown models are returned as-is."""
    assert resolve_model_id("some/unknown-model", hub="hf") == "some/unknown-model"
    assert resolve_model_id("my-custom-model", hub="ms") == "my-custom-model"


def test_resolve_all_registry_models():
    """Every model in the registry resolves successfully on both hubs."""
    for name, entry in MODEL_REGISTRY.items():
        ms_id = resolve_model_id(name, hub="ms")
        hf_id = resolve_model_id(name, hub="hf")
        assert ms_id == entry["ms"], f"{name}: ms expected {entry['ms']}, got {ms_id}"
        assert hf_id == entry["hf"], f"{name}: hf expected {entry['hf']}, got {hf_id}"


def test_resolve_default_hub_is_ms():
    """Default hub parameter is 'ms'."""
    result = resolve_model_id("SenseVoiceSmall")
    assert result == "iic/SenseVoiceSmall"


# ------------------------------------------------------------------
# MODEL_REGISTRY structure
# ------------------------------------------------------------------

def test_registry_has_required_keys():
    """Every registry entry has ms, hf, and type keys."""
    for name, entry in MODEL_REGISTRY.items():
        assert "ms" in entry, f"{name} missing 'ms' key"
        assert "hf" in entry, f"{name} missing 'hf' key"
        assert "type" in entry, f"{name} missing 'type' key"


def test_registry_model_types():
    """Registry contains models of expected types."""
    types = {entry["type"] for entry in MODEL_REGISTRY.values()}
    assert "asr" in types
    assert "vad" in types
    assert "punc" in types
    assert "spk" in types


def test_registry_has_common_models():
    """Registry includes the most commonly used models."""
    assert "SenseVoiceSmall" in MODEL_REGISTRY
    assert "fsmn-vad" in MODEL_REGISTRY
    assert "ct-punc" in MODEL_REGISTRY
    assert "cam++" in MODEL_REGISTRY


# ------------------------------------------------------------------
# list_available_models
# ------------------------------------------------------------------

def test_list_available_models():
    models = list_available_models()
    assert isinstance(models, dict)
    assert len(models) == len(MODEL_REGISTRY)
    assert "SenseVoiceSmall" in models
    assert models["SenseVoiceSmall"]["type"] == "asr"


def test_list_available_models_returns_copy():
    """list_available_models returns a copy, not the original."""
    models = list_available_models()
    models["SenseVoiceSmall"]["type"] = "modified"
    assert MODEL_REGISTRY["SenseVoiceSmall"]["type"] == "asr"
