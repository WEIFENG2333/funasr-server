"""FunASR Server — client SDK for host applications.

Supports ALL FunASR model types: ASR, VAD, punctuation, speaker,
emotion, alignment, keyword spotting, etc.

Usage:
    from funasr_server import FunASR

    asr = FunASR()
    asr.ensure_installed()
    asr.start()

    # ASR — model name auto-resolved to correct hub
    asr.load_model(model="SenseVoiceSmall")
    result = asr.infer(audio="audio.wav", language="zh", use_itn=True)

    # VAD (standalone)
    asr.load_model(model="fsmn-vad", name="vad")
    result = asr.infer(audio="audio.wav", name="vad")

    # ASR + VAD pipeline
    asr.load_model(model="SenseVoiceSmall", vad_model="fsmn-vad", name="asr_vad")
    result = asr.infer(audio="audio.wav", name="asr_vad")

    # Punctuation model
    asr.load_model(model="ct-punc", name="punc")
    result = asr.infer(text="你好世界今天天气真好", name="punc")

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
from funasr_server.mirror import get_hub
from funasr_server.models import resolve_model_id

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
        hub: str = None,
        batch_size: int = None,
        quantize: bool = None,
        fp16: bool = None,
        disable_update: bool = None,
        **kwargs,
    ) -> dict:
        """Load any FunASR model via AutoModel.

        Works with ALL model types: ASR, VAD, punctuation, speaker,
        emotion, alignment, keyword spotting, etc.

        Model names are automatically resolved to the correct hub-specific
        ID based on the detected network region. For example,
        ``load_model(model="SenseVoiceSmall")`` will automatically use
        ``iic/SenseVoiceSmall`` in China or ``FunAudioLLM/SenseVoiceSmall``
        internationally.

        Args:
            model: Model name or ID (e.g. "SenseVoiceSmall", "fsmn-vad").
            name: Cache key for this model instance.
            vad_model: VAD model for ASR pipeline composition.
            punc_model: Punctuation model for ASR pipeline.
            spk_model: Speaker model for ASR pipeline.
            device: "cuda" / "cpu" / None (auto-detect).
            hub: "ms" (ModelScope) / "hf" (HuggingFace) / None (auto-detect).
            batch_size: Inference batch size.
            quantize: Enable model quantization.
            fp16: Enable half-precision inference.
            disable_update: Skip model update checks on startup.
            **kwargs: Additional AutoModel parameters. Common options:
                - ncpu (int): Number of CPU threads.
                - ngpu (int): Number of GPUs.
                - seed (int): Random seed.
                - trust_remote_code (bool): Allow remote code execution.
                - remote_code (str): Path to custom model code.
                - model_revision (str): Model version string.
                - vad_kwargs (dict): Extra VAD model parameters.
                - punc_kwargs (dict): Extra punctuation model parameters.
                - spk_kwargs (dict): Extra speaker model parameters.

        Returns:
            {"name": str, "status": "loaded" | "already_loaded"}
        """
        if hub is None:
            hub = get_hub()

        resolved_model = resolve_model_id(model, hub=hub)
        logger.info(f"Resolved model '{model}' -> '{resolved_model}' (hub={hub})")

        # Build params, filtering out None values from named params
        named = {
            "model": resolved_model,
            "hub": hub,
            "vad_model": resolve_model_id(vad_model, hub=hub) if vad_model else None,
            "punc_model": resolve_model_id(punc_model, hub=hub) if punc_model else None,
            "spk_model": resolve_model_id(spk_model, hub=hub) if spk_model else None,
            "device": device,
            "batch_size": batch_size,
            "quantize": quantize,
            "fp16": fp16,
            "disable_update": disable_update,
        }
        params = {k: v for k, v in named.items() if v is not None}
        params["name"] = name  # always include name
        params.update(kwargs)

        return self._rpc_call("load_model", params, timeout=600)

    def unload_model(self, name: str = "default") -> dict:
        """Unload a model and free memory."""
        return self._rpc_call("unload_model", {"name": name})

    def infer(
        self,
        audio: str = None,
        text: str = None,
        audio_bytes: bytes = None,
        name: str = "default",
        language: str = None,
        use_itn: bool = None,
        batch_size: int = None,
        hotword: str = None,
        merge_vad: bool = None,
        merge_length_s: float = None,
        output_timestamp: bool = None,
        **kwargs,
    ) -> list:
        """Universal inference — works with ANY loaded FunASR model.

        Calls model.generate(input=..., **kwargs) on the server side.
        Provide exactly one of: audio, text, or audio_bytes.

        Args:
            audio: Path to audio file (for ASR/VAD/speaker models).
            text: Text string (for punctuation/text models).
            audio_bytes: Raw audio bytes (WAV/MP3/etc.).
            name: Model cache key (default: "default").
            language: Language code (e.g. "zh", "en", "ja").
            use_itn: Enable inverse text normalization.
            batch_size: Inference batch size.
            hotword: Hotword file path or hotword string.
            merge_vad: Merge short VAD segments.
            merge_length_s: Max merge length in seconds.
            output_timestamp: Include timestamps in output.
            **kwargs: Additional generate() parameters. Common options:
                - itn (bool): Inverse text normalization (some models).
                - text_norm (str): Text normalization mode.
                - batch_size_s (int): Batch size in seconds for VAD inference.
                - data_type (str): Input data type hint.

        Returns:
            List of result dicts. Structure depends on model type:
            - ASR: [{"key": ..., "text": ...}]
            - VAD: [{"key": ..., "value": [[start_ms, end_ms], ...]}]
            - Punctuation: [{"key": ..., "text": ...}]
        """
        # Build generate kwargs, filtering out None values
        named_generate = {
            "language": language,
            "use_itn": use_itn,
            "batch_size": batch_size,
            "hotword": hotword,
            "merge_vad": merge_vad,
            "merge_length_s": merge_length_s,
            "output_timestamp": output_timestamp,
        }
        params = {"name": name}
        params.update({k: v for k, v in named_generate.items() if v is not None})
        params.update(kwargs)

        # Set input
        if audio_bytes is not None:
            params["input_base64"] = base64.b64encode(audio_bytes).decode()
        elif audio is not None:
            if os.path.exists(audio):
                params["input"] = str(Path(audio).resolve())
            else:
                params["input"] = audio
        elif text is not None:
            params["input"] = text
        else:
            raise ValueError(
                "Provide exactly one of: 'audio' (file path), "
                "'text' (text string), or 'audio_bytes' (raw bytes)"
            )

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
            **kwargs: Additional generate() parameters
                (language, use_itn, hotword, etc.)

        Returns:
            List of result dicts with at least "text" key.
        """
        return self.infer(
            audio=audio,
            audio_bytes=audio_bytes,
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
        """Download a model to local cache.

        Args:
            model: Model name or ID (e.g. "SenseVoiceSmall",
                "iic/SenseVoiceSmall").
            hub: "ms" (ModelScope) / "hf" (HuggingFace) / None (auto-detect).

        Returns:
            {"model": str, "path": str, "hub": str}
        """
        if hub is None:
            hub = get_hub()
        resolved = resolve_model_id(model, hub=hub)
        return self._rpc_call("download_model", {"model": resolved, "hub": hub}, timeout=600)

    def list_models(self) -> dict:
        """List all loaded models."""
        return self._rpc_call("list_models", {})

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _rpc_call(self, method: str, params: dict, timeout: float = 30) -> Any:
        """Send a JSON-RPC 2.0 request to the server.

        Validates the response strictly per the JSON-RPC 2.0 spec.

        Raises:
            ConnectionError: Network error or malformed response.
            ServerError: Server returned a JSON-RPC error.
        """
        self._rpc_id += 1
        request_id = self._rpc_id
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
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
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot connect to server at {url}: {e}")

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConnectionError(f"Invalid JSON response from server: {e}")

        # Validate JSON-RPC 2.0 response structure
        if not isinstance(body, dict):
            raise ConnectionError("Invalid JSON-RPC response: expected JSON object")

        if body.get("jsonrpc") != "2.0":
            raise ConnectionError(
                f"Invalid JSON-RPC response: expected jsonrpc='2.0', "
                f"got {body.get('jsonrpc')!r}"
            )

        if body.get("id") != request_id:
            raise ConnectionError(
                f"JSON-RPC response ID mismatch: "
                f"expected {request_id}, got {body.get('id')!r}"
            )

        # Check for error response
        if "error" in body:
            err = body["error"]
            if not isinstance(err, dict) or "code" not in err or "message" not in err:
                raise ConnectionError(f"Malformed JSON-RPC error object: {err!r}")
            raise ServerError(err["code"], err["message"], err.get("data"))

        # Must have result
        if "result" not in body:
            raise ConnectionError(
                "Invalid JSON-RPC response: missing 'result' field"
            )

        return body["result"]

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
