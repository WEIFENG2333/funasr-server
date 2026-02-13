"""Mirror source auto-detection for pip/PyTorch/ModelScope.

Detects whether the user is in China (mainland) by testing pypi.org
connectivity, then returns appropriate mirror URLs.
"""

import logging
import socket

logger = logging.getLogger(__name__)

_CN_PIP_INDEX = "https://mirrors.aliyun.com/pypi/simple/"
_CN_PIP_TRUSTED_HOST = "mirrors.aliyun.com"
_CN_TORCH_INDEX = "https://mirror.sjtu.edu.cn/pytorch-wheels/cu121"
_CN_TORCH_CPU_INDEX = "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu"

_INTL_TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
_INTL_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


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

    Returns:
        "cn" or "intl"
    """
    if _test_connectivity("pypi.org", 443, timeout=3.0):
        logger.info("Network: international (pypi.org reachable)")
        return "intl"
    else:
        logger.info("Network: China mainland (pypi.org unreachable, using mirrors)")
        return "cn"


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
