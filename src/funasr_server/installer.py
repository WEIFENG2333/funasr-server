"""Runtime environment installer.

Handles:
1. Detecting/installing uv
2. Creating runtime directory with pyproject.toml + server.py
3. Running uv sync to install all dependencies
"""

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

from funasr_server.mirror import detect_region, get_uv_env

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
        """Create runtime directory and copy template files."""
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        for src_file in _TEMPLATE_DIR.iterdir():
            dst_file = self.runtime_dir / src_file.name
            if src_file.is_file():
                shutil.copy2(src_file, dst_file)
                logger.info(f"Copied {src_file.name} -> {dst_file}")

        models_dir = self.runtime_dir / "models"
        models_dir.mkdir(exist_ok=True)

    def _uv_sync(self):
        """Run uv sync to install all dependencies."""
        uv = self.get_uv_path()
        if not uv:
            raise RuntimeError("uv not available")

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
