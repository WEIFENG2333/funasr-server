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
    assert config["pip_index_url"] is not None
    assert "aliyun" in config["pip_index_url"]
    assert config["model_hub"] == "ms"


def test_get_mirror_config_intl():
    config = get_mirror_config("intl")
    assert config["pip_index_url"] is None
    assert config["model_hub"] == "hf"


def test_get_uv_env_cn():
    env = get_uv_env("cn")
    assert "UV_INDEX_URL" in env
    assert "aliyun" in env["UV_INDEX_URL"]


def test_get_uv_env_intl():
    env = get_uv_env("intl")
    assert env == {}


def test_connectivity_unreachable():
    result = _test_connectivity("192.0.2.1", 1, timeout=0.1)
    assert result is False
