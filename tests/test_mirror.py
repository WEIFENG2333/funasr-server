"""Tests for mirror detection and configuration."""

from unittest.mock import patch

from funasr_server.mirror import (
    _test_connectivity,
    detect_region,
    get_mirror_config,
    get_uv_env,
)


def test_detect_region_international():
    with patch("funasr_server.mirror._test_connectivity", return_value=True):
        assert detect_region() == "intl"


def test_detect_region_china():
    with patch("funasr_server.mirror._test_connectivity", return_value=False):
        assert detect_region() == "cn"


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

    with patch("funasr_server.mirror.detect_region", return_value="intl"):
        config = get_mirror_config(None)
        assert config["model_hub"] == "hf"


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

    with patch("funasr_server.mirror.detect_region", return_value="intl"):
        env = get_uv_env(None)
        assert env == {}


def test_connectivity_unreachable():
    result = _test_connectivity("192.0.2.1", 1, timeout=0.1)
    assert result is False


def test_connectivity_os_error():
    """OSError (not just timeout) also returns False."""
    with patch("funasr_server.mirror.socket.create_connection", side_effect=OSError("refused")):
        result = _test_connectivity("example.com", 443, timeout=1.0)
        assert result is False
