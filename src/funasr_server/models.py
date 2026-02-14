"""Model name registry for FunASR.

Maps user-friendly model names to hub-specific model IDs so that users
can use simple names like "SenseVoiceSmall" regardless of which hub
(ModelScope or HuggingFace) they are using.

Usage:
    from funasr_server.models import resolve_model_id, MODEL_REGISTRY

    # Returns the correct hub-specific model ID
    model_id = resolve_model_id("SenseVoiceSmall", hub="hf")
    # → "FunAudioLLM/SenseVoiceSmall"

    model_id = resolve_model_id("SenseVoiceSmall", hub="ms")
    # → "iic/SenseVoiceSmall"

    # Short aliases also work
    model_id = resolve_model_id("fsmn-vad", hub="hf")
    # → "fsmn-vad"  (FunASR internally resolves short aliases)
"""

import logging

logger = logging.getLogger(__name__)


# Model registry: name → {"ms": model_id, "hf": model_id, "type": str}
#
# FunASR has its own name_maps for short aliases like "fsmn-vad", "ct-punc",
# "cam++" etc. Those work on both hubs without us doing anything.
#
# This registry handles models that DON'T have short aliases (like
# SenseVoiceSmall) or where users might use various names.
MODEL_REGISTRY = {
    # ASR models
    "SenseVoiceSmall": {
        "ms": "iic/SenseVoiceSmall",
        "hf": "FunAudioLLM/SenseVoiceSmall",
        "type": "asr",
    },
    "paraformer-zh": {
        "ms": "paraformer-zh",
        "hf": "paraformer-zh",
        "type": "asr",
    },
    "paraformer-en": {
        "ms": "paraformer-en",
        "hf": "paraformer-en",
        "type": "asr",
    },
    "Whisper-large-v3": {
        "ms": "Whisper-large-v3",
        "hf": "Whisper-large-v3",
        "type": "asr",
    },
    "Whisper-large-v3-turbo": {
        "ms": "Whisper-large-v3-turbo",
        "hf": "Whisper-large-v3-turbo",
        "type": "asr",
    },

    # VAD models
    "fsmn-vad": {
        "ms": "fsmn-vad",
        "hf": "fsmn-vad",
        "type": "vad",
    },

    # Punctuation models
    "ct-punc": {
        "ms": "ct-punc",
        "hf": "ct-punc",
        "type": "punc",
    },

    # Speaker models
    "cam++": {
        "ms": "cam++",
        "hf": "cam++",
        "type": "spk",
    },

    # Timestamp / forced alignment
    "fa-zh": {
        "ms": "fa-zh",
        "hf": "fa-zh",
        "type": "fa",
    },

    # Emotion models
    "emotion2vec_plus_large": {
        "ms": "emotion2vec_plus_large",
        "hf": "emotion2vec_plus_large",
        "type": "emotion",
    },
}

# Case-insensitive + alias lookup table (built once)
_ALIAS_MAP = {}


def _build_alias_map():
    """Build a case-insensitive alias lookup table."""
    if _ALIAS_MAP:
        return
    for name, entry in MODEL_REGISTRY.items():
        _ALIAS_MAP[name.lower()] = name
        # Also add hub-specific IDs as aliases
        for hub_id in (entry.get("ms"), entry.get("hf")):
            if hub_id and hub_id.lower() != name.lower():
                _ALIAS_MAP[hub_id.lower()] = name


def resolve_model_id(model: str, hub: str = "ms") -> str:
    """Resolve a model name to a hub-specific model ID.

    Lookup order:
    1. Exact match in MODEL_REGISTRY
    2. Case-insensitive match
    3. Match by hub-specific ID (reverse lookup)
    4. Return the original name unchanged (FunASR may resolve it internally)

    Args:
        model: Model name or ID. Can be a simple name ("SenseVoiceSmall"),
               a short alias ("fsmn-vad"), or a full ID ("iic/SenseVoiceSmall").
        hub: "ms" for ModelScope, "hf" for HuggingFace.

    Returns:
        The hub-specific model ID string.
    """
    _build_alias_map()

    # Exact match
    if model in MODEL_REGISTRY:
        return MODEL_REGISTRY[model].get(hub, model)

    # Case-insensitive lookup
    canonical = _ALIAS_MAP.get(model.lower())
    if canonical:
        return MODEL_REGISTRY[canonical].get(hub, model)

    # Not in our registry — return as-is (FunASR handles its own aliases)
    return model


def list_available_models() -> dict:
    """List all models in the registry.

    Returns:
        Dict mapping model names to their info (type, hub IDs).
    """
    return {name: dict(entry) for name, entry in MODEL_REGISTRY.items()}
