"""FunASR Server — client SDK for host applications.

Supports ALL FunASR model types: ASR, VAD, punctuation, speaker,
emotion, alignment, keyword spotting, etc.

Usage:
    from funasr_server import FunASR

    asr = FunASR()
    asr.ensure_installed()
    asr.start()

    # ASR
    asr.load_model(model="iic/SenseVoiceSmall")
    result = asr.infer("audio.wav", language="zh", use_itn=True)

    # VAD (standalone)
    asr.load_model(model="fsmn-vad", name="vad")
    result = asr.infer("audio.wav", name="vad")

    # Punctuation (text input)
    asr.load_model(model="ct-punc", name="punc")
    result = asr.infer("hello world", name="punc")

    asr.stop()
"""

import base64
import json
import logging
import os
import platform
import signal
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Optional

from funasr_server.installer import Installer

logger = logging.getLogger(__name__)


class ServerError(Exception):
    """Error returned by the FunASR server."""
    def __init__(self, code: int, message: str, data: str = None):
        self.code = code
        self.data = data
        super().__init__(message)


class FunASR:
    """Client for the FunASR inference server.

    Manages installation, lifecycle, and communication with
    a background FunASR server process.

    Args:
        runtime_dir: Directory for the server runtime environment.
        port: Port for the server. 0 = auto-assign.
        host: Bind host (default: 127.0.0.1).
    """

    def __init__(
        self,
        runtime_dir: str = "./funasr_runtime",
        port: int = 0,
        host: str = "127.0.0.1",
    ):
        self.runtime_dir = Path(runtime_dir).resolve()
        self.port = port
        self.host = host
        self.installer = Installer(str(self.runtime_dir))
        self._process: Optional[subprocess.Popen] = None
        self._rpc_id = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_installed(self, progress_callback=None) -> bool:
        """Ensure runtime environment is installed.

        Args:
            progress_callback: optional callable(step: str, detail: str)

        Returns:
            True if already installed, False if fresh install was performed.
        """
        if self.installer.is_installed():
            logger.info("Runtime environment already installed")
            return True

        logger.info("Runtime environment not found, installing...")
        self.installer.install(progress_callback=progress_callback)
        return False

    def start(self, timeout: float = 60) -> int:
        """Start the server as a background process.

        Args:
            timeout: Max seconds to wait for server to become ready.

        Returns:
            The port number the server is listening on.
        """
        if self._process and self._process.poll() is None:
            logger.info(f"Server already running (pid={self._process.pid})")
            return self.port

        uv_path = self.installer.get_uv_path()
        if not uv_path:
            raise RuntimeError("uv not found. Call ensure_installed() first.")

        server_py = str(self.runtime_dir / "server.py")
        cmd = [
            uv_path, "run",
            "--project", str(self.runtime_dir),
            "python", server_py,
            "--host", self.host,
            "--port", str(self.port),
        ]

        logger.info(f"Starting server: {' '.join(cmd)}")
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.runtime_dir),
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self._process.stdout.readline()
            if not line:
                if self._process.poll() is not None:
                    stderr = self._process.stderr.read().decode(errors="replace")
                    raise RuntimeError(f"Server failed to start:\n{stderr}")
                continue

            line = line.decode().strip()
            logger.debug(f"Server output: {line}")
            if line.startswith("PLUGIN_PORT="):
                self.port = int(line.split("=")[1])
                logger.info(f"Server started on port {self.port} (pid={self._process.pid})")
                break
        else:
            self.stop()
            raise TimeoutError(f"Server did not start within {timeout}s")

        self._wait_for_ready(timeout=max(5, timeout - (time.time() - start_time)))
        return self.port

    def stop(self):
        """Stop the server process."""
        if self._process is None:
            return

        try:
            self._rpc_call("shutdown", {}, timeout=3)
        except Exception:
            pass

        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not exit gracefully, killing...")
            if platform.system() == "Windows":
                self._process.terminate()
            else:
                self._process.send_signal(signal.SIGKILL)
            self._process.wait(timeout=5)

        self._process = None
        logger.info("Server stopped")

    def is_running(self) -> bool:
        """Check if the server process is alive."""
        if self._process is None:
            return False
        return self._process.poll() is None

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def health(self) -> dict:
        """Check server health."""
        return self._rpc_call("health", {})

    def load_model(
        self,
        model: str,
        name: str = "default",
        vad_model: str = None,
        punc_model: str = None,
        spk_model: str = None,
        device: str = None,
        **kwargs,
    ) -> dict:
        """Load any FunASR model via AutoModel.

        Works with ALL model types: ASR, VAD, punctuation, speaker,
        emotion, alignment, keyword spotting, etc.

        Args:
            model: Model ID (e.g. "iic/SenseVoiceSmall", "fsmn-vad", "ct-punc")
            name: Cache key for this model instance.
            vad_model: VAD model ID (only for ASR pipeline composition).
            punc_model: Punctuation model ID (only for ASR pipeline).
            spk_model: Speaker model ID (only for ASR pipeline).
            device: "cuda" / "cpu" / None (auto).
            **kwargs: Additional AutoModel parameters.
        """
        params = {"model": model, "name": name, **kwargs}
        if vad_model:
            params["vad_model"] = vad_model
        if punc_model:
            params["punc_model"] = punc_model
        if spk_model:
            params["spk_model"] = spk_model
        if device:
            params["device"] = device
        return self._rpc_call("load_model", params, timeout=300)

    def unload_model(self, name: str = "default") -> dict:
        """Unload a model and free memory."""
        return self._rpc_call("unload_model", {"name": name})

    def infer(
        self,
        input: str = None,
        input_bytes: bytes = None,
        name: str = "default",
        **kwargs,
    ) -> list:
        """Universal inference — works with ANY loaded FunASR model.

        Calls model.generate(input=..., **kwargs) on the server side.

        Args:
            input: File path (audio) or text string (punctuation models).
            input_bytes: Raw audio bytes as alternative to file path.
            name: Model cache key (default: "default").
            **kwargs: Additional generate() parameters (language, use_itn, etc.)

        Returns:
            List of result dicts. Structure depends on model type:
            - ASR: [{"key": ..., "text": ...}]
            - VAD: [{"key": ..., "value": [[start_ms, end_ms], ...]}]
            - Punctuation: [{"key": ..., "text": ..., "punc_array": ...}]
        """
        params = {"name": name, **kwargs}

        if input_bytes:
            params["input_base64"] = base64.b64encode(input_bytes).decode()
        elif input is not None:
            if os.path.exists(input):
                params["input"] = str(Path(input).resolve())
            else:
                params["input"] = input
        else:
            raise ValueError("Either 'input' or 'input_bytes' must be provided")

        result = self._rpc_call("infer", params, timeout=600)
        return result.get("results", [])

    def transcribe(
        self,
        audio: str = None,
        audio_bytes: bytes = None,
        name: str = "default",
        **kwargs,
    ) -> list:
        """Transcribe audio — convenience alias for infer().

        Args:
            audio: Path to audio file.
            audio_bytes: Raw audio bytes (WAV/MP3/etc.)
            name: Model cache key.
            **kwargs: Additional generate() parameters.

        Returns:
            List of result dicts with at least "text" key.
        """
        return self.infer(
            input=audio,
            input_bytes=audio_bytes,
            name=name,
            **kwargs,
        )

    def execute(self, code: str, return_var: str = "result", **kwargs) -> dict:
        """Execute arbitrary Python code in the server environment.

        Args:
            code: Python code string to execute.
            return_var: Variable name whose value to return.

        Returns:
            {"output": stdout_str, "return_value": value, "error": error_or_None}
        """
        params = {"code": code, "return_var": return_var, **kwargs}
        return self._rpc_call("execute", params, timeout=600)

    def download_model(self, model: str, hub: str = None) -> dict:
        """Download a model.

        Args:
            model: Model ID (e.g. "iic/SenseVoiceSmall")
            hub: "ms" (ModelScope) or "hf" (HuggingFace).
        """
        params = {"model": model}
        if hub:
            params["hub"] = hub
        return self._rpc_call("download_model", params, timeout=600)

    def list_models(self) -> dict:
        """List all loaded models."""
        return self._rpc_call("list_models", {})

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _rpc_call(self, method: str, params: dict, timeout: float = 30) -> Any:
        """Send a JSON-RPC 2.0 request to the server."""
        self._rpc_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._rpc_id,
            "method": method,
            "params": params,
        }

        url = f"http://{self.host}:{self.port}/rpc"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot connect to server at {url}: {e}")

        if "error" in body:
            err = body["error"]
            raise ServerError(err["code"], err["message"], err.get("data"))

        return body.get("result")

    def _wait_for_ready(self, timeout: float = 30):
        """Wait for the HTTP server to accept connections."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                self.health()
                return
            except Exception:
                time.sleep(0.3)
        raise TimeoutError(f"Server not responding after {timeout}s")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.ensure_installed()
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def __del__(self):
        try:
            if self._process and self._process.poll() is None:
                self.stop()
        except Exception:
            pass
