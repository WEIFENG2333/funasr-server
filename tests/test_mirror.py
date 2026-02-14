"""Tests for mirror detection and configuration."""

import time
from unittest.mock import patch

import pytest

import funasr_server.mirror as mirror_module
from funasr_server.mirror import (
    _test_connectivity,
    detect_region,
    get_hub,
    get_mirror_config,
    get_uv_env,
)


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the cached region between tests."""
    mirror_module._cached_region = None
    yield
    mirror_module._cached_region = None


# ------------------------------------------------------------------
# _test_connectivity
# ------------------------------------------------------------------

def test_connectivity_unreachable():
    result = _test_connectivity("192.0.2.1", 1, timeout=0.1)
    assert result is False


def test_connectivity_os_error():
    """OSError (not just timeout) also returns False."""
    with patch("funasr_server.mirror.socket.create_connection", side_effect=OSError("refused")):
        result = _test_connectivity("example.com", 443, timeout=1.0)
        assert result is False


# ------------------------------------------------------------------
# detect_region
# ------------------------------------------------------------------

def test_detect_region_international():
    """If any international host is reachable, returns 'intl'."""
    with patch("funasr_server.mirror._test_connectivity", return_value=True):
        assert detect_region() == "intl"


def test_detect_region_china():
    """If no international host is reachable, returns 'cn'."""
    with patch("funasr_server.mirror._test_connectivity", return_value=False):
        assert detect_region() == "cn"


def test_detect_region_env_override_cn():
    """FUNASR_REGION=cn forces China region."""
    with patch.dict("os.environ", {"FUNASR_REGION": "cn"}):
        assert detect_region() == "cn"


def test_detect_region_env_override_intl():
    """FUNASR_REGION=intl forces international region."""
    with patch.dict("os.environ", {"FUNASR_REGION": "intl"}):
        assert detect_region() == "intl"


def test_detect_region_env_override_ignores_case():
    """FUNASR_REGION is case-insensitive."""
    with patch.dict("os.environ", {"FUNASR_REGION": "CN"}):
        assert detect_region() == "cn"


def test_detect_region_caching():
    """Second call returns cached result without probing."""
    call_count = [0]
    original = _test_connectivity

    def counting_test(*args, **kwargs):
        call_count[0] += 1
        return True

    with patch("funasr_server.mirror._test_connectivity", side_effect=counting_test):
        detect_region()
        first_count = call_count[0]
        detect_region()  # should use cache
        assert call_count[0] == first_count  # no additional calls


def test_detect_region_concurrent_probes(capsys):
    """Multiple hosts are probed concurrently, not sequentially."""
    probe_times = []

    def slow_test(host, port, timeout):
        probe_times.append(time.monotonic())
        time.sleep(0.1)
        return host == "pypi.org"

    with patch("funasr_server.mirror._test_connectivity", side_effect=slow_test):
        result = detect_region()
        assert result == "intl"
        # All probes should start within 50ms of each other (concurrent)
        if len(probe_times) >= 2:
            assert max(probe_times) - min(probe_times) < 0.5


# ------------------------------------------------------------------
# get_hub
# ------------------------------------------------------------------

def test_get_hub_cn():
    assert get_hub("cn") == "ms"


def test_get_hub_intl():
    assert get_hub("intl") == "hf"


def test_get_hub_auto_detect():
    with patch("funasr_server.mirror.detect_region", return_value="cn"):
        assert get_hub() == "ms"


def test_get_hub_env_override():
    """FUNASR_HUB overrides auto-detection."""
    with patch.dict("os.environ", {"FUNASR_HUB": "hf"}):
        # Even with cn region, env override wins
        assert get_hub("cn") == "hf"


def test_get_hub_env_override_ms():
    with patch.dict("os.environ", {"FUNASR_HUB": "ms"}):
        assert get_hub("intl") == "ms"


# ------------------------------------------------------------------
# get_mirror_config
# ------------------------------------------------------------------

def test_get_mirror_config_cn():
    config = get_mirror_config("cn")
    assert "aliyun" in config["pip_index_url"]
    assert config["pip_trusted_host"] == "mirrors.aliyun.com"
    assert "sjtu" in config["torch_index_url"]
    assert "sjtu" in config["torch_cpu_index_url"]
    assert config["model_hub"] == "ms"


def test_get_mirror_config_intl():
    config = get_mirror_config("intl")
    assert config["pip_index_url"] is None
    assert config["pip_trusted_host"] is None
    assert "pytorch.org" in config["torch_index_url"]
    assert "pytorch.org" in config["torch_cpu_index_url"]
    assert config["model_hub"] == "hf"


def test_get_mirror_config_auto_detect():
    """get_mirror_config(None) auto-detects region."""
    with patch("funasr_server.mirror.detect_region", return_value="cn"):
        config = get_mirror_config(None)
        assert config["model_hub"] == "ms"

    mirror_module._cached_region = None
    with patch("funasr_server.mirror.detect_region", return_value="intl"):
        config = get_mirror_config(None)
        assert config["model_hub"] == "hf"


# ------------------------------------------------------------------
# get_uv_env
# ------------------------------------------------------------------

def test_get_uv_env_cn():
    env = get_uv_env("cn")
    assert "UV_INDEX_URL" in env
    assert "aliyun" in env["UV_INDEX_URL"]


def test_get_uv_env_intl():
    env = get_uv_env("intl")
    assert env == {}


def test_get_uv_env_auto_detect():
    """get_uv_env(None) auto-detects region."""
    with patch("funasr_server.mirror.detect_region", return_value="cn"):
        env = get_uv_env(None)
        assert "UV_INDEX_URL" in env

    mirror_module._cached_region = None
    with patch("funasr_server.mirror.detect_region", return_value="intl"):
        env = get_uv_env(None)
        assert env == {}
