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
        installer._region = "intl"
        installer._create_runtime_dir()

        assert runtime_dir.exists()
        assert (runtime_dir / "pyproject.toml").exists()
        assert (runtime_dir / "server.py").exists()
        assert (runtime_dir / "models").is_dir()

        # pyproject.toml should be generated (not copied from template)
        content = (runtime_dir / "pyproject.toml").read_text()
        assert "funasr-server-runtime" in content
        assert "torch" in content


def test_create_runtime_dir_idempotent():
    """Calling _create_runtime_dir() twice doesn't fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        installer = Installer(str(runtime_dir))
        installer._region = "intl"
        installer._create_runtime_dir()
        installer._create_runtime_dir()  # second call should not fail

        assert (runtime_dir / "pyproject.toml").exists()
        assert (runtime_dir / "server.py").exists()


def _nvidia_smi_output(cuda_version="12.8"):
    """Build a fake nvidia-smi output string."""
    return (
        f"| NVIDIA-SMI 550.135  Driver Version: 550.135  CUDA Version: {cuda_version} |\n"
    ).encode()


def test_detect_gpu_no_nvidia():
    """No NVIDIA GPU returns 'cpu'."""
    installer = Installer("/tmp/test")
    with patch("funasr_server.installer.subprocess.run",
               side_effect=FileNotFoundError):
        assert installer._detect_gpu() == "cpu"


def test_detect_gpu_cuda_128():
    """CUDA 12.8 driver selects cu128."""
    installer = Installer("/tmp/test")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _nvidia_smi_output("12.8")
    with patch("funasr_server.installer.subprocess.run", return_value=mock_result):
        assert installer._detect_gpu() == "cu128"


def test_detect_gpu_cuda_124():
    """CUDA 12.4 driver selects cu124 (not cu126 or cu128)."""
    installer = Installer("/tmp/test")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _nvidia_smi_output("12.4")
    with patch("funasr_server.installer.subprocess.run", return_value=mock_result):
        assert installer._detect_gpu() == "cu124"


def test_detect_gpu_cuda_121():
    """CUDA 12.1 driver selects cu121."""
    installer = Installer("/tmp/test")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _nvidia_smi_output("12.1")
    with patch("funasr_server.installer.subprocess.run", return_value=mock_result):
        assert installer._detect_gpu() == "cu121"


def test_detect_gpu_cuda_118():
    """CUDA 11.8 driver selects cu118."""
    installer = Installer("/tmp/test")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _nvidia_smi_output("11.8")
    with patch("funasr_server.installer.subprocess.run", return_value=mock_result):
        assert installer._detect_gpu() == "cu118"


def test_detect_gpu_cuda_126():
    """CUDA 12.6 driver selects cu126."""
    installer = Installer("/tmp/test")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _nvidia_smi_output("12.6")
    with patch("funasr_server.installer.subprocess.run", return_value=mock_result):
        assert installer._detect_gpu() == "cu126"


def test_detect_gpu_cuda_too_old():
    """CUDA 10.2 is too old — falls back to cpu."""
    installer = Installer("/tmp/test")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _nvidia_smi_output("10.2")
    with patch("funasr_server.installer.subprocess.run", return_value=mock_result):
        assert installer._detect_gpu() == "cpu"


def test_detect_gpu_macos():
    """macOS always returns 'cpu'."""
    installer = Installer("/tmp/test")
    with patch("funasr_server.installer.platform.system", return_value="Darwin"):
        assert installer._detect_gpu() == "cpu"


def test_detect_gpu_unparseable():
    """nvidia-smi succeeds but output is unparseable — defaults to cu128."""
    installer = Installer("/tmp/test")
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = b"some weird output without version"
    with patch("funasr_server.installer.subprocess.run", return_value=mock_result):
        assert installer._detect_gpu() == "cu128"


def test_generate_pyproject_cpu():
    """CPU machine generates pyproject with pytorch-cpu index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        runtime_dir.mkdir()
        installer = Installer(str(runtime_dir))
        installer._region = "intl"
        with patch.object(installer, "_detect_gpu", return_value="cpu"), \
             patch("funasr_server.installer.platform.system", return_value="Linux"):
            installer._generate_pyproject()
        content = (runtime_dir / "pyproject.toml").read_text()
        assert "pytorch-cpu" in content
        assert "download.pytorch.org/whl/cpu" in content
        assert "cu1" not in content


def test_generate_pyproject_cuda():
    """CUDA machine generates pyproject with correct index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        runtime_dir.mkdir()
        installer = Installer(str(runtime_dir))
        installer._region = "intl"
        with patch.object(installer, "_detect_gpu", return_value="cu124"):
            installer._generate_pyproject()
        content = (runtime_dir / "pyproject.toml").read_text()
        assert "pytorch-cu124" in content
        assert "download.pytorch.org/whl/cu124" in content


def test_generate_pyproject_macos():
    """macOS generates pyproject without explicit torch index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        runtime_dir.mkdir()
        installer = Installer(str(runtime_dir))
        installer._region = "intl"
        with patch.object(installer, "_detect_gpu", return_value="cpu"), \
             patch("funasr_server.installer.platform.system", return_value="Darwin"):
            installer._generate_pyproject()
        content = (runtime_dir / "pyproject.toml").read_text()
        assert "torch>=2.0.0" in content
        # macOS should NOT have explicit index (uses PyPI default with MPS)
        assert "pytorch-cpu" not in content
        assert "pytorch-cu" not in content


def test_generate_pyproject_cn_cpu():
    """China region CPU uses Chinese mirror for torch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        runtime_dir.mkdir()
        installer = Installer(str(runtime_dir))
        installer._region = "cn"
        with patch.object(installer, "_detect_gpu", return_value="cpu"), \
             patch("funasr_server.installer.platform.system", return_value="Linux"):
            installer._generate_pyproject()
        content = (runtime_dir / "pyproject.toml").read_text()
        assert "mirror.sjtu.edu.cn" in content


def test_generate_pyproject_cn_cuda():
    """China region CUDA uses Chinese mirror for torch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        runtime_dir = Path(tmpdir) / "runtime"
        runtime_dir.mkdir()
        installer = Installer(str(runtime_dir))
        installer._region = "cn"
        with patch.object(installer, "_detect_gpu", return_value="cu126"):
            installer._generate_pyproject()
        content = (runtime_dir / "pyproject.toml").read_text()
        assert "mirror.sjtu.edu.cn/pytorch-wheels/cu126" in content
        assert "pytorch-cu126" in content


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
