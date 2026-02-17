"""Runtime environment installer.

Handles:
1. Detecting/installing uv
2. Creating runtime directory with pyproject.toml + server.py
3. Detecting GPU to choose correct PyTorch variant (CUDA vs CPU)
4. Running uv sync to install all dependencies
"""

import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from funasr_server.mirror import detect_region, get_mirror_config, get_uv_env

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "runtime_template"


class Installer:
    def __init__(self, runtime_dir: str):
        self.runtime_dir = Path(runtime_dir).resolve()
        self._uv_path = None
        self._region = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_installed(self) -> bool:
        """Check if the runtime environment is already installed."""
        venv = self.runtime_dir / ".venv"
        pyproject = self.runtime_dir / "pyproject.toml"
        server = self.runtime_dir / "server.py"
        return venv.exists() and pyproject.exists() and server.exists()

    def install(self, progress_callback=None):
        """Full installation: uv + runtime dir + dependencies.

        Args:
            progress_callback: optional callable(step: str, detail: str)
        """
        def _progress(step, detail=""):
            logger.info(f"[install] {step}: {detail}")
            if progress_callback:
                progress_callback(step, detail)

        _progress("detect_region", "Detecting network region...")
        self._region = detect_region()
        _progress("detect_region", f"Region: {self._region}")

        _progress("ensure_uv", "Checking uv installation...")
        self._ensure_uv()
        _progress("ensure_uv", f"uv ready: {self._uv_path}")

        _progress("create_runtime", "Setting up runtime directory...")
        self._create_runtime_dir()
        _progress("create_runtime", f"Runtime dir: {self.runtime_dir}")

        _progress("uv_sync", "Installing dependencies (this may take a few minutes)...")
        self._uv_sync()
        _progress("uv_sync", "Dependencies installed successfully")

    def get_uv_path(self) -> str:
        """Return path to uv binary."""
        if self._uv_path:
            return self._uv_path
        self._uv_path = shutil.which("uv")
        if self._uv_path:
            return self._uv_path
        # Check common install locations
        home = Path.home()
        candidates = [
            home / ".local" / "bin" / "uv",
            home / ".cargo" / "bin" / "uv",
            Path(os.environ.get("LOCALAPPDATA", "")) / "uv" / "uv.exe",
            home / ".local" / "bin" / "uv.exe",
        ]
        for c in candidates:
            if c.exists():
                self._uv_path = str(c)
                return self._uv_path
        return None

    def get_python_path(self) -> str:
        """Return path to the Python inside the runtime's .venv."""
        if platform.system() == "Windows":
            return str(self.runtime_dir / ".venv" / "Scripts" / "python.exe")
        return str(self.runtime_dir / ".venv" / "bin" / "python")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_uv(self):
        """Install uv if not already available."""
        if self.get_uv_path():
            logger.info(f"uv found at: {self._uv_path}")
            return

        logger.info("uv not found, installing...")
        system = platform.system()

        if system in ("Linux", "Darwin"):
            cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            subprocess.run(cmd, shell=True, check=True)
        elif system == "Windows":
            cmd = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
            subprocess.run(cmd, shell=True, check=True)
        else:
            raise RuntimeError(f"Unsupported platform: {system}")

        if not self.get_uv_path():
            raise RuntimeError(
                "uv installation completed but binary not found. "
                "Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
            )

    def _create_runtime_dir(self):
        """Create runtime directory and copy template files.

        Copies all template files except pyproject.toml, which is
        dynamically generated based on GPU detection and region.
        """
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        for src_file in _TEMPLATE_DIR.iterdir():
            if src_file.name == "pyproject.toml":
                continue  # generated dynamically below
            dst_file = self.runtime_dir / src_file.name
            if src_file.is_file():
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied {src_file.name} -> {dst_file}")

        self._generate_pyproject()

        models_dir = self.runtime_dir / "models"
        models_dir.mkdir(exist_ok=True)

    # PyTorch CUDA index versions available on download.pytorch.org,
    # ordered from newest to oldest.  _detect_gpu() picks the newest
    # index whose CUDA version is <= the driver's CUDA version.
    _CUDA_INDEXES = ["cu128", "cu126", "cu124", "cu121", "cu118"]

    def _detect_gpu(self) -> str:
        """Detect GPU type and CUDA version for PyTorch variant selection.

        Returns:
            "cu128", "cu126", etc. for NVIDIA GPU, or "cpu" if no GPU.
            macOS always returns "cpu" (PyTorch uses MPS from default PyPI).
        """
        if platform.system() == "Darwin":
            logger.info("GPU detection: macOS — using PyPI default (MPS support built-in)")
            return "cpu"

        # Check for NVIDIA GPU via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                raise FileNotFoundError
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("GPU detection: no NVIDIA GPU — using CPU-only PyTorch")
            return "cpu"

        # Parse CUDA version from nvidia-smi output
        # Example line: "NVIDIA-SMI 550.135  Driver Version: 550.135  CUDA Version: 12.4"
        output = result.stdout.decode(errors="replace")
        match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
        if not match:
            logger.warning("GPU detection: nvidia-smi found but couldn't parse CUDA version, using cu128")
            return "cu128"

        cuda_major = int(match.group(1))
        cuda_minor = int(match.group(2))
        driver_cuda = cuda_major * 10 + cuda_minor  # e.g. 12.4 -> 124
        logger.info(f"GPU detection: NVIDIA CUDA {cuda_major}.{cuda_minor} (driver)")

        # Find the best matching PyTorch CUDA index
        # Pick the newest index whose version is <= driver CUDA version
        for index in self._CUDA_INDEXES:
            index_version = int(index[2:])  # "cu128" -> 128
            if index_version <= driver_cuda:
                logger.info(f"GPU detection: selected PyTorch index {index}")
                return index

        # Driver CUDA is older than all available indexes
        logger.warning(
            f"GPU detection: CUDA {cuda_major}.{cuda_minor} is too old for "
            f"available PyTorch builds (need >= 11.8), falling back to CPU"
        )
        return "cpu"

    def _generate_pyproject(self):
        """Generate pyproject.toml with correct PyTorch index for this machine."""
        gpu = self._detect_gpu()
        region = self._region or "intl"
        mirror = get_mirror_config(region)

        base_deps = [
            '"funasr @ git+https://github.com/modelscope/FunASR.git"',
            '"modelscope"',
            '"huggingface_hub"',
            '"transformers"',
            '"tiktoken"',
            '"torch>=2.0.0"',
            '"torchaudio>=2.0.0"',
            '"uvicorn>=0.30.0"',
            '"starlette>=0.37.0"',
        ]

        lines = [
            "[project]",
            'name = "funasr-server-runtime"',
            'version = "0.1.0"',
            'requires-python = ">=3.10,<3.13"',
            "dependencies = [",
        ]
        for dep in base_deps:
            lines.append(f"    {dep},")
        lines.append("]")
        lines.append("")

        if gpu == "cpu" and platform.system() != "Darwin":
            # CPU-only: use PyTorch CPU index
            torch_url = f"{mirror['torch_base_url']}/cpu"
            index_name = "pytorch-cpu"
            lines.extend([
                "[[tool.uv.index]]",
                f'name = "{index_name}"',
                f'url = "{torch_url}"',
                "explicit = true",
                "",
                "[tool.uv.sources]",
                f'torch = [{{ index = "{index_name}" }}]',
                f'torchaudio = [{{ index = "{index_name}" }}]',
            ])
            logger.info(f"PyTorch config: CPU-only (index: {torch_url})")

        elif gpu.startswith("cu"):
            # CUDA: use matching PyTorch CUDA index
            torch_url = f"{mirror['torch_base_url']}/{gpu}"
            index_name = f"pytorch-{gpu}"
            lines.extend([
                "[[tool.uv.index]]",
                f'name = "{index_name}"',
                f'url = "{torch_url}"',
                "explicit = true",
                "",
                "[tool.uv.sources]",
                f'torch = [{{ index = "{index_name}" }}]',
                f'torchaudio = [{{ index = "{index_name}" }}]',
            ])
            logger.info(f"PyTorch config: {gpu.upper()} (index: {torch_url})")

        else:
            # macOS: use default PyPI (has MPS support built-in)
            logger.info("PyTorch config: default PyPI (macOS)")

        lines.append("")

        pyproject_path = self.runtime_dir / "pyproject.toml"
        pyproject_path.write_text("\n".join(lines))
        logger.info(f"Generated {pyproject_path}")

    def _uv_sync(self):
        """Run uv sync to install all dependencies.

        Deletes any existing uv.lock to force re-resolution, since
        the pyproject.toml is generated per-machine based on GPU detection.
        """
        uv = self.get_uv_path()
        if not uv:
            raise RuntimeError("uv not available")

        # Remove stale lock file to force fresh resolution
        lock_file = self.runtime_dir / "uv.lock"
        if lock_file.exists():
            lock_file.unlink()
            logger.info("Removed stale uv.lock for fresh resolution")

        env = os.environ.copy()
        env.update(get_uv_env(self._region))

        cmd = [uv, "sync", "--project", str(self.runtime_dir)]

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(self.runtime_dir),
            capture_output=False,
        )

        if result.returncode != 0:
            raise RuntimeError(f"uv sync failed with return code {result.returncode}")
