"""Mirror source auto-detection for pip/PyTorch/ModelScope.

Detects whether the user is in China (mainland) or international by
testing connectivity to multiple well-known hosts, then returns
appropriate mirror URLs and model hub preference.

Override auto-detection with environment variables:
    FUNASR_HUB=ms   → force ModelScope
    FUNASR_HUB=hf   → force HuggingFace
    FUNASR_REGION=cn / FUNASR_REGION=intl → force region
"""

import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

_CN_PIP_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
_CN_PIP_TRUSTED_HOST = "mirrors.aliyun.com"
_CN_TORCH_INDEX = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"
_CN_TORCH_CPU_INDEX = "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu"

_INTL_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
_INTL_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

# International hosts to probe — if ANY of these are reachable, we're international
_INTL_HOSTS = [
    ("huggingface.co", 443),
    ("pypi.org", 443),
    ("github.com", 443),
]

# Cache to avoid re-detecting every call
_cached_region = None


def _test_connectivity(host: str = "pypi.org", port: int = 443, timeout: float = 3.0) -> bool:
    """Test if a host is reachable within timeout."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (socket.timeout, OSError):
        return False


def detect_region() -> str:
    """Detect whether the user is in China or international.

    Detection strategy:
    1. Check FUNASR_REGION environment variable (explicit override).
    2. Probe multiple international hosts concurrently (huggingface.co,
       pypi.org, github.com). If ANY responds, return "intl".
    3. Otherwise return "cn".

    Results are cached for the process lifetime.

    Returns:
        "cn" or "intl"
    """
    global _cached_region

    # Environment variable override
    env_region = os.environ.get("FUNASR_REGION", "").lower().strip()
    if env_region in ("cn", "intl"):
        logger.info(f"Network region override via FUNASR_REGION={env_region}")
        _cached_region = env_region
        return env_region

    # Return cached result
    if _cached_region is not None:
        return _cached_region

    # Probe multiple hosts concurrently — return "intl" as soon as any succeeds
    with ThreadPoolExecutor(max_workers=len(_INTL_HOSTS)) as pool:
        futures = {
            pool.submit(_test_connectivity, host, port, 3.0): host
            for host, port in _INTL_HOSTS
        }
        for future in as_completed(futures, timeout=5.0):
            host = futures[future]
            try:
                if future.result():
                    logger.info(f"Network: international ({host} reachable)")
                    _cached_region = "intl"
                    return "intl"
            except Exception:
                pass

    logger.info("Network: China mainland (international hosts unreachable, using mirrors)")
    _cached_region = "cn"
    return "cn"


def get_hub(region: str = None) -> str:
    """Get the preferred model hub for the given region.

    Args:
        region: "cn" or "intl". If None, auto-detect.

    Returns:
        "ms" for ModelScope (China) or "hf" for HuggingFace (international).
    """
    # Explicit hub override takes highest priority
    env_hub = os.environ.get("FUNASR_HUB", "").lower().strip()
    if env_hub in ("ms", "hf"):
        return env_hub

    if region is None:
        region = detect_region()
    return "ms" if region == "cn" else "hf"


def get_mirror_config(region: str = None) -> dict:
    """Get mirror configuration based on region.

    Args:
        region: "cn" or "intl". If None, auto-detect.

    Returns:
        dict with keys: pip_index_url, pip_trusted_host,
        torch_index_url, torch_cpu_index_url, model_hub
    """
    if region is None:
        region = detect_region()

    if region == "cn":
        return {
            "pip_index_url": _CN_PIP_INDEX,
            "pip_trusted_host": _CN_PIP_TRUSTED_HOST,
            "torch_index_url": _CN_TORCH_INDEX,
            "torch_cpu_index_url": _CN_TORCH_CPU_INDEX,
            "model_hub": "ms",
        }
    else:
        return {
            "pip_index_url": None,
            "pip_trusted_host": None,
            "torch_index_url": _INTL_TORCH_INDEX,
            "torch_cpu_index_url": _INTL_TORCH_CPU_INDEX,
            "model_hub": "hf",
        }


def get_uv_env(region: str = None) -> dict:
    """Get environment variables to pass to uv commands for mirror config.

    Args:
        region: "cn" or "intl". If None, auto-detect.

    Returns:
        dict of environment variable overrides
    """
    config = get_mirror_config(region)
    env = {}
    if config["pip_index_url"]:
        env["UV_INDEX_URL"] = config["pip_index_url"]
    return env
