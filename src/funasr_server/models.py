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

    # Any FunASR model ID works even if not in the registry
    model_id = resolve_model_id("iic/some-custom-model", hub="ms")
    # → "iic/some-custom-model"  (passed through unchanged)
"""

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry: name → {"ms": model_id, "hf": model_id, "type": str}
#
# For models that have short aliases in FunASR's own name_maps
# (fsmn-vad, ct-punc, cam++, etc.), the short name works on both hubs
# so ms/hf values are identical.
#
# For models WITHOUT short aliases (SenseVoiceSmall, Fun-ASR-Nano, etc.),
# we provide the correct hub-specific full ID.
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    # =================================================================
    # ASR — SenseVoice
    # =================================================================
    "SenseVoiceSmall": {
        "ms": "iic/SenseVoiceSmall",
        "hf": "FunAudioLLM/SenseVoiceSmall",
        "type": "asr",
        "desc": "Multi-task ASR (zh/en/ja/ko/yue), 234M params",
    },

    # =================================================================
    # ASR — Fun-ASR-Nano (latest 2025 models)
    # =================================================================
    "Fun-ASR-Nano": {
        "ms": "FunAudioLLM/Fun-ASR-Nano-2512",
        "hf": "FunAudioLLM/Fun-ASR-Nano-2512",
        "type": "asr",
        "desc": "End-to-end ASR, zh (7 dialects, 26 accents) + en + ja, 800M params",
    },
    "Fun-ASR-MLT-Nano": {
        "ms": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "hf": "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
        "type": "asr",
        "desc": "Multilingual ASR (31 languages), 800M params",
    },

    # =================================================================
    # ASR — Paraformer family
    # =================================================================
    "paraformer": {
        "ms": "paraformer",
        "hf": "paraformer",
        "type": "asr",
        "desc": "Paraformer-large, zh+en, 220M, offline (max 20s)",
    },
    "paraformer-zh": {
        "ms": "paraformer-zh",
        "hf": "paraformer-zh",
        "type": "asr",
        "desc": "Paraformer-large with SeACo, zh+en, 220M, arbitrary length",
    },
    "paraformer-en": {
        "ms": "paraformer-en",
        "hf": "paraformer-en",
        "type": "asr",
        "desc": "Paraformer-large, English, 220M, long audio",
    },
    "paraformer-en-spk": {
        "ms": "paraformer-en-spk",
        "hf": "paraformer-en-spk",
        "type": "asr",
        "desc": "Paraformer-large + speaker diarization, 220M",
    },
    "paraformer-zh-streaming": {
        "ms": "paraformer-zh-streaming",
        "hf": "paraformer-zh-streaming",
        "type": "asr",
        "desc": "Paraformer-large streaming, zh+en, 220M, online",
    },

    # =================================================================
    # ASR — Whisper (via FunASR wrapper)
    # =================================================================
    "Whisper-large-v2": {
        "ms": "Whisper-large-v2",
        "hf": "Whisper-large-v2",
        "type": "asr",
        "desc": "OpenAI Whisper large-v2, multilingual",
    },
    "Whisper-large-v3": {
        "ms": "Whisper-large-v3",
        "hf": "Whisper-large-v3",
        "type": "asr",
        "desc": "OpenAI Whisper large-v3, multilingual, 1550M",
    },
    "Whisper-large-v3-turbo": {
        "ms": "Whisper-large-v3-turbo",
        "hf": "Whisper-large-v3-turbo",
        "type": "asr",
        "desc": "OpenAI Whisper large-v3-turbo, multilingual, 809M",
    },

    # =================================================================
    # ASR — Qwen-Audio
    # =================================================================
    "Qwen-Audio": {
        "ms": "Qwen-Audio",
        "hf": "Qwen-Audio",
        "type": "asr",
        "desc": "Qwen-Audio multimodal, 8B params",
    },

    # =================================================================
    # VAD — Voice Activity Detection
    # =================================================================
    "fsmn-vad": {
        "ms": "fsmn-vad",
        "hf": "fsmn-vad",
        "type": "vad",
        "desc": "FSMN-VAD, 0.4M params, 16kHz",
    },

    # =================================================================
    # Punctuation Restoration
    # =================================================================
    "ct-punc": {
        "ms": "ct-punc",
        "hf": "ct-punc",
        "type": "punc",
        "desc": "CT-Transformer punctuation, zh+en, 1.1G (large)",
    },
    "ct-punc-c": {
        "ms": "ct-punc-c",
        "hf": "ct-punc-c",
        "type": "punc",
        "desc": "CT-Transformer punctuation, zh+en, 291M",
    },

    # =================================================================
    # Speaker Verification / Embedding
    # =================================================================
    "cam++": {
        "ms": "cam++",
        "hf": "cam++",
        "type": "spk",
        "desc": "CAM++ speaker embedding, 7.2M params",
    },

    # =================================================================
    # Timestamp / Forced Alignment
    # =================================================================
    "fa-zh": {
        "ms": "fa-zh",
        "hf": "fa-zh",
        "type": "fa",
        "desc": "Timestamp prediction, zh, 37.8M params",
    },

    # =================================================================
    # Emotion Recognition
    # =================================================================
    "emotion2vec_plus_large": {
        "ms": "emotion2vec_plus_large",
        "hf": "emotion2vec_plus_large",
        "type": "emotion",
        "desc": "Emotion recognition, 300M params, 5 emotions",
    },
    "emotion2vec_plus_base": {
        "ms": "emotion2vec_plus_base",
        "hf": "emotion2vec_plus_base",
        "type": "emotion",
        "desc": "Emotion recognition (base), 5 emotions",
    },
    "emotion2vec_plus_seed": {
        "ms": "emotion2vec_plus_seed",
        "hf": "emotion2vec_plus_seed",
        "type": "emotion",
        "desc": "Emotion recognition (seed), 5 emotions",
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
        Dict mapping model names to their info (type, hub IDs, description).
    """
    return {name: dict(entry) for name, entry in MODEL_REGISTRY.items()}
