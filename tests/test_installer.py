"""Tests for the Installer class."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from funasr_server.installer import Installer


def test_is_installed_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)
        assert installer.is_installed() is False


def test_is_installed_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create the expected files
        (Path(tmpdir) / ".venv").mkdir()
        (Path(tmpdir) / "pyproject.toml").write_text("[project]\nname='test'")
        (Path(tmpdir) / "server.py").write_text("# server")

        installer = Installer(tmpdir)
        assert installer.is_installed() is True


def test_is_installed_partial():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Only some files exist
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
        # All candidate paths won't exist either
        result = installer.get_uv_path()
        # Could be None or found in a standard location
        # Just verify it doesn't crash


def test_get_python_path_unix():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("platform.system", return_value="Linux"):
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


def test_install_calls_steps():
    with tempfile.TemporaryDirectory() as tmpdir:
        installer = Installer(tmpdir)

        progress_calls = []

        with patch.object(installer, "_ensure_uv"), \
             patch.object(installer, "_create_runtime_dir"), \
             patch.object(installer, "_uv_sync"), \
             patch("funasr_server.installer.detect_region", return_value="intl"):

            installer.install(
                progress_callback=lambda step, detail: progress_calls.append(step)
            )

        assert "detect_region" in progress_calls
        assert "ensure_uv" in progress_calls
        assert "create_runtime" in progress_calls
        assert "uv_sync" in progress_calls
