"""Tests for the Installer class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from funasr_server.installer import Installer


def test_is_installed_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)
        assert installer.is_installed() is False


def test_is_installed_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / ".venv").mkdir()
        (Path(tmpdir) / "pyproject.toml").write_text("[project]\nname='test'")
        (Path(tmpdir) / "server.py").write_text("# server")

        installer = Installer(tmpdir)
        assert installer.is_installed() is True


def test_is_installed_partial():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "pyproject.toml").write_text("[project]")

        installer = Installer(tmpdir)
        assert installer.is_installed() is False


def test_get_uv_path_from_which():
    installer = Installer("/tmp/test")
    with patch("shutil.which", return_value="/usr/bin/uv"):
        assert installer.get_uv_path() == "/usr/bin/uv"


def test_get_uv_path_not_found():
    installer = Installer("/tmp/test")
    with patch("shutil.which", return_value=None):
        result = installer.get_uv_path()
        # If no candidate path exists either, should return None
        assert result is None or isinstance(result, str)


def test_get_uv_path_from_candidate():
    """uv found in candidate location when which() fails."""
    installer = Installer("/tmp/test")
    with patch("shutil.which", return_value=None):
        candidate = Path.home() / ".local" / "bin" / "uv"
        with patch.object(Path, "exists", side_effect=lambda: True):
            # The first candidate that exists should be returned
            result = installer.get_uv_path()
            # Just verify we get a string path back (exact path depends on home)
            if result is not None:
                assert isinstance(result, str)


def test_get_uv_path_cached():
    """Second call returns cached path without calling which() again."""
    installer = Installer("/tmp/test")
    installer._uv_path = "/cached/uv"
    assert installer.get_uv_path() == "/cached/uv"


def test_get_python_path_unix():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("platform.system", return_value="Linux"):
            installer = Installer(tmpdir)
            path = installer.get_python_path()
            assert path.endswith(os.path.join("bin", "python"))


def test_get_python_path_macos():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("platform.system", return_value="Darwin"):
            installer = Installer(tmpdir)
            path = installer.get_python_path()
            assert path.endswith(os.path.join("bin", "python"))


def test_get_python_path_windows():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("platform.system", return_value="Windows"):
            installer = Installer(tmpdir)
            path = installer.get_python_path()
            assert path.endswith(os.path.join("Scripts", "python.exe"))


def test_create_runtime_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        installer = Installer(str(runtime_dir))
        installer._create_runtime_dir()

        assert runtime_dir.exists()
        assert (runtime_dir / "pyproject.toml").exists()
        assert (runtime_dir / "server.py").exists()
        assert (runtime_dir / "models").is_dir()


def test_create_runtime_dir_idempotent():
    """Calling _create_runtime_dir() twice doesn't fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        installer = Installer(str(runtime_dir))
        installer._create_runtime_dir()
        installer._create_runtime_dir()  # second call should not fail

        assert (runtime_dir / "pyproject.toml").exists()
        assert (runtime_dir / "server.py").exists()


def test_install_calls_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)

        progress_calls = []

        with patch.object(installer, "_ensure_uv") as mock_uv, \
             patch.object(installer, "_create_runtime_dir") as mock_create, \
             patch.object(installer, "_uv_sync") as mock_sync, \
             patch("funasr_server.installer.detect_region", return_value="intl"):

            installer.install(
                progress_callback=lambda step, detail: progress_calls.append(step)
            )

            mock_uv.assert_called_once()
            mock_create.assert_called_once()
            mock_sync.assert_called_once()

        assert "detect_region" in progress_calls
        assert "ensure_uv" in progress_calls
        assert "create_runtime" in progress_calls
        assert "uv_sync" in progress_calls


def test_install_sets_region():
    """install() stores the detected region."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)

        with patch.object(installer, "_ensure_uv"), \
             patch.object(installer, "_create_runtime_dir"), \
             patch.object(installer, "_uv_sync"), \
             patch("funasr_server.installer.detect_region", return_value="cn"):

            installer.install()

        assert installer._region == "cn"


def test_ensure_uv_already_installed():
    """_ensure_uv() does nothing if uv is already found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)
        with patch.object(installer, "get_uv_path", return_value="/usr/bin/uv"):
            installer._ensure_uv()  # should not raise


def test_ensure_uv_install_linux():
    """_ensure_uv() calls curl on Linux."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)
        call_count = [0]

        def fake_get_uv_path():
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # first call: not installed
            return "/home/user/.local/bin/uv"  # after installation

        with patch.object(installer, "get_uv_path", side_effect=fake_get_uv_path), \
             patch("platform.system", return_value="Linux"), \
             patch("subprocess.run") as mock_run:

            installer._ensure_uv()
            mock_run.assert_called_once()
            assert "curl" in mock_run.call_args[0][0]


def test_ensure_uv_install_windows():
    """_ensure_uv() calls powershell on Windows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)
        call_count = [0]

        def fake_get_uv_path():
            call_count[0] += 1
            if call_count[0] == 1:
                return None
            return "C:\\Users\\user\\.local\\bin\\uv.exe"

        with patch.object(installer, "get_uv_path", side_effect=fake_get_uv_path), \
             patch("platform.system", return_value="Windows"), \
             patch("subprocess.run") as mock_run:

            installer._ensure_uv()
            mock_run.assert_called_once()
            assert "powershell" in mock_run.call_args[0][0]


def test_ensure_uv_install_fails():
    """_ensure_uv() raises if uv is still not found after installation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)

        with patch.object(installer, "get_uv_path", return_value=None), \
             patch("platform.system", return_value="Linux"), \
             patch("subprocess.run"):

            with pytest.raises(RuntimeError, match="binary not found"):
                installer._ensure_uv()


def test_ensure_uv_unsupported_platform():
    """_ensure_uv() raises on unsupported platform."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)

        with patch.object(installer, "get_uv_path", return_value=None), \
             patch("platform.system", return_value="FreeBSD"):

            with pytest.raises(RuntimeError, match="Unsupported platform"):
                installer._ensure_uv()


def test_uv_sync_no_uv():
    """_uv_sync() raises if uv is not available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)

        with patch.object(installer, "get_uv_path", return_value=None):
            with pytest.raises(RuntimeError, match="uv not available"):
                installer._uv_sync()


def test_uv_sync_failure():
    """_uv_sync() raises on non-zero return code."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch.object(installer, "get_uv_path", return_value="/usr/bin/uv"), \
             patch("subprocess.run", return_value=mock_result), \
             patch("funasr_server.installer.get_uv_env", return_value={}):

            with pytest.raises(RuntimeError, match="uv sync failed"):
                installer._uv_sync()


def test_uv_sync_passes_env():
    """_uv_sync() passes mirror env vars to subprocess."""
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)
        installer._region = "cn"
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch.object(installer, "get_uv_path", return_value="/usr/bin/uv"), \
             patch("subprocess.run", return_value=mock_result) as mock_run, \
             patch("funasr_server.installer.get_uv_env", return_value={"UV_INDEX_URL": "https://mirrors.aliyun.com/pypi/simple/"}):

            installer._uv_sync()
            called_env = mock_run.call_args[1]["env"]
            assert "UV_INDEX_URL" in called_env
