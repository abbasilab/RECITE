"""
llm.py

vLLM client and prompt management.
"""
import json
import os
import signal
import sys
import subprocess
import threading
import time
import httpx
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional
from loguru import logger

# config/ is at project root, this file is in recite/crawler/
PROMPTS_PATH = Path(__file__).parent.parent.parent / "config" / "crawler_prompts.json"


def _get_cache_dir() -> Path:
    """Get the cache directory for vLLM server configuration.
    
    Returns:
        Path to `.clintrialm_cache/` directory in project root.
        Creates the directory if it doesn't exist.
    """
    cache_dir = Path(__file__).parent.parent.parent / ".clintrialm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def check_server_health(endpoint: str, timeout: int = 5) -> bool:
    """Check if vLLM server is responding.
    
    Args:
        endpoint: Base endpoint URL (e.g., "http://localhost:8000/v1")
        timeout: Request timeout in seconds
        
    Returns:
        True if server responds with 200, False otherwise
    """
    try:
        client = httpx.Client(timeout=timeout)
        resp = client.get(f"{endpoint}/models", timeout=timeout)
        client.close()
        return resp.status_code == 200
    except (httpx.RequestError, httpx.HTTPStatusError, Exception):
        return False


def get_vllm_server_pid(cache_dir: Path | None = None) -> int | None:
    """Get the PID of the running vLLM server from cache.
    
    Args:
        cache_dir: Cache directory path. If None, uses default from _get_cache_dir()
        
    Returns:
        PID if process is running, None otherwise
    """
    if cache_dir is None:
        cache_dir = _get_cache_dir()
    
    pid_file = cache_dir / "vllm.pid"
    
    if not pid_file.exists():
        return None
    
    try:
        pid = int(pid_file.read_text().strip())
        
        # Check if process is still running
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
            return pid
        except OSError:
            # Process doesn't exist, clean up stale PID file
            pid_file.unlink()
            return None
    except (ValueError, OSError):
        # Invalid PID or file read error
        return None


def stop_vllm_server(cache_dir: Path | None = None) -> bool:
    """Stop the running vLLM server and watchdog (if running).
    
    Args:
        cache_dir: Cache directory path. If None, uses default from _get_cache_dir()
        
    Returns:
        True if server was stopped successfully, False otherwise
    """
    if cache_dir is None:
        cache_dir = _get_cache_dir()
    
    # Check for watchdog PID (supervised mode)
    watchdog_pid_file = cache_dir / "serve_watchdog.pid"
    watchdog_pid = None
    if watchdog_pid_file.exists():
        try:
            watchdog_pid = int(watchdog_pid_file.read_text().strip())
            # Check if process is still running
            try:
                os.kill(watchdog_pid, 0)
            except OSError:
                # Process doesn't exist, clean up stale PID file
                watchdog_pid_file.unlink()
                watchdog_pid = None
        except (ValueError, OSError):
            watchdog_pid_file.unlink()
            watchdog_pid = None
    
    # Get vLLM PID
    vllm_pid = get_vllm_server_pid(cache_dir)
    
    if vllm_pid is None and watchdog_pid is None:
        logger.warning("No running vLLM server or watchdog found (no valid PID files)")
        return False
    
    stopped_any = False
    
    # Stop vLLM first
    if vllm_pid is not None:
        try:
            os.kill(vllm_pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to vLLM server (PID {vllm_pid}), waiting for graceful shutdown...")
            
            # Wait up to 10 seconds for process to exit
            for _ in range(10):
                time.sleep(1)
                try:
                    os.kill(vllm_pid, 0)  # Check if still running
                except OSError:
                    # Process has exited
                    pid_file = cache_dir / "vllm.pid"
                    if pid_file.exists():
                        pid_file.unlink()
                    logger.info(f"vLLM server (PID {vllm_pid}) stopped gracefully")
                    stopped_any = True
                    break
            
            if not stopped_any:
                # Process still running, force kill with SIGKILL
                logger.warning(f"Server did not stop gracefully, sending SIGKILL to PID {vllm_pid}")
                os.kill(vllm_pid, signal.SIGKILL)
                time.sleep(1)
                pid_file = cache_dir / "vllm.pid"
                if pid_file.exists():
                    pid_file.unlink()
                logger.info(f"vLLM server (PID {vllm_pid}) force-stopped")
                stopped_any = True
                
        except OSError as e:
            logger.error(f"Failed to stop vLLM server (PID {vllm_pid}): {e}")
            pid_file = cache_dir / "vllm.pid"
            if pid_file.exists():
                pid_file.unlink()
    
    # Stop watchdog (supervised serve process)
    if watchdog_pid is not None:
        try:
            os.kill(watchdog_pid, signal.SIGTERM)
            logger.info(f"Sent SIGTERM to watchdog (PID {watchdog_pid}), waiting for graceful shutdown...")
            
            # Wait up to 10 seconds
            for _ in range(10):
                time.sleep(1)
                try:
                    os.kill(watchdog_pid, 0)
                except OSError:
                    # Process has exited
                    if watchdog_pid_file.exists():
                        watchdog_pid_file.unlink()
                    logger.info(f"Watchdog (PID {watchdog_pid}) stopped gracefully")
                    stopped_any = True
                    break
            
            if not stopped_any:
                logger.warning(f"Watchdog did not stop gracefully, sending SIGKILL to PID {watchdog_pid}")
                os.kill(watchdog_pid, signal.SIGKILL)
                time.sleep(1)
                if watchdog_pid_file.exists():
                    watchdog_pid_file.unlink()
                logger.info(f"Watchdog (PID {watchdog_pid}) force-stopped")
                stopped_any = True
                
        except OSError as e:
            logger.error(f"Failed to stop watchdog (PID {watchdog_pid}): {e}")
            if watchdog_pid_file.exists():
                watchdog_pid_file.unlink()
    
    return stopped_any


def wait_for_server_ready(endpoint: str, max_wait_seconds: int = 300, poll_interval: int = 2) -> bool:
    """Wait for vLLM server to become ready by polling health check.
    
    Args:
        endpoint: Server endpoint (e.g., "http://localhost:8000/v1")
        max_wait_seconds: Maximum time to wait for server to become ready
        poll_interval: Seconds between health check attempts
        
    Returns:
        True if server became ready, False if timeout
    """
    start_time = time.time()
    max_polls = max_wait_seconds // poll_interval
    
    logger.info(f"Waiting for vLLM server to become ready at {endpoint}...")
    
    for attempt in range(max_polls):
        if check_server_health(endpoint, timeout=5):
            elapsed = time.time() - start_time
            logger.info(f"vLLM server is ready after {elapsed:.1f}s")
            return True
        
        time.sleep(poll_interval)
        if attempt % 5 == 0:  # Log every 10 seconds
            elapsed = time.time() - start_time
            logger.info(f"Waiting for vLLM server to be ready... ({elapsed:.1f}s elapsed)")
    
    # Server didn't become ready in time
    elapsed = time.time() - start_time
    logger.error(f"vLLM server did not become ready within {max_wait_seconds}s (waited {elapsed:.1f}s)")
    return False


def _build_vllm_command(config: dict) -> list[str]:
    """Build vLLM command from configuration.
    
    Args:
        config: Configuration dictionary with model, port, tensor_parallel, etc.
        
    Returns:
        List of command arguments for subprocess
    """
    model_name = config["model"]
    port = config.get("port", 8000)
    tp = config.get("tensor_parallel", 1)
    gpu_mem = config.get("gpu_mem", 0.9)
    max_len = config.get("max_model_len")
    enforce_eager = config.get("enforce_eager", False)
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", str(gpu_mem),
        "--trust-remote-code",
    ]
    if max_len:
        cmd.extend(["--max-model-len", str(max_len)])
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    return cmd


def _build_vllm_env(config: dict) -> dict[str, str]:
    """Build environment variables for vLLM process.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Environment dictionary
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if config.get("cuda_visible_devices"):
        env["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]
    if config.get("hf_home"):
        env["HF_HOME"] = config["hf_home"]
        env["TRANSFORMERS_CACHE"] = config["hf_home"]
    return env


def load_vllm_config(cache_dir: Path | None = None) -> dict | None:
    """Load vLLM server configuration from cache.
    
    Args:
        cache_dir: Cache directory path. If None, uses default from _get_cache_dir()
        
    Returns:
        Configuration dictionary, or None if config file doesn't exist or is invalid
    """
    if cache_dir is None:
        cache_dir = _get_cache_dir()
    
    config_file = cache_dir / "vllm_serve_config.json"
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load vLLM server configuration: {e}")
        return None


def save_vllm_config(cache_dir: Path, config_data: dict) -> None:
    """Save vLLM server configuration to cache.
    
    Args:
        cache_dir: Cache directory path
        config_data: Configuration dictionary to save
    """
    config_file = cache_dir / "vllm_serve_config.json"
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)


def _start_vllm_process(
    cmd: list[str],
    env: dict[str, str],
    cache_dir: Path,
    config_data: dict,
    stream_logs_to_terminal: bool = False,
) -> Optional[subprocess.Popen]:
    """Start vLLM process with given command and environment.
    
    Args:
        cmd: Command to run (list of strings)
        env: Environment variables
        cache_dir: Cache directory for PID and log files
        config_data: Configuration data to update with PID/log paths
        stream_logs_to_terminal: If True, use inherited stdout/stderr. If False, use log files.
        
    Returns:
        subprocess.Popen object, or None if startup failed
    """
    import subprocess
    from datetime import datetime, timezone
    
    try:
        model_name = config_data.get("model", "unknown")
        port = config_data.get("port", 8000)
        logger.info(f"Starting vLLM server: {model_name} on port {port}")
        
        if stream_logs_to_terminal:
            # Use inherited stdout/stderr for terminal streaming
            stdout = None
            stderr = None
            start_new_session = False  # Keep in same process group for Ctrl+C handling
        else:
            # Create log files for background mode
            log_dir = cache_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stdout_log = log_dir / f"vllm_stdout_{timestamp}.log"
            stderr_log = log_dir / f"vllm_stderr_{timestamp}.log"
            
            stdout_file = open(stdout_log, "w")
            stderr_file = open(stderr_log, "w")
            stdout = stdout_file
            stderr = stderr_file
            start_new_session = True  # Detach from terminal for background mode
            
            # Store log paths in config
            config_data["stdout_log"] = str(stdout_log)
            config_data["stderr_log"] = str(stderr_log)
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=start_new_session,
        )
        
        # Close file handles in parent if using log files
        if not stream_logs_to_terminal:
            stdout_file.close()
            stderr_file.close()
        
        # Save PID
        pid_file = cache_dir / "vllm.pid"
        pid_file.write_text(str(process.pid))
        
        # Update config with PID and timestamp
        config_data["pid"] = process.pid
        config_data["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        save_vllm_config(cache_dir, config_data)
        
        logger.info(f"vLLM server started (PID: {process.pid})")
        return process
        
    except Exception as e:
        logger.error(f"Failed to start vLLM server process: {e}")
        return None


def ensure_server_ready(endpoint: str, cache_dir: Path | None = None, max_wait_seconds: int = 300) -> bool:
    """Ensure vLLM server is ready, waiting if it's starting or restarting if needed.
    
    This function checks:
    1. If server is already ready (health check passes) - returns immediately
    2. If server process exists but not ready - waits for it to become ready
    3. If no server process exists - restarts the server and waits
    
    Args:
        endpoint: Server endpoint (e.g., "http://localhost:8000/v1")
        cache_dir: Cache directory path. If None, uses default from _get_cache_dir()
        max_wait_seconds: Maximum time to wait for server to become ready
        
    Returns:
        True if server is ready, False otherwise
    """
    if cache_dir is None:
        cache_dir = _get_cache_dir()
    
    # First check if server is already ready
    if check_server_health(endpoint, timeout=5):
        logger.debug("vLLM server is already ready")
        return True
    
    # Check if server process exists (might be starting)
    pid = get_vllm_server_pid(cache_dir)
    if pid is not None:
        # Server process exists but not ready yet - wait for it
        logger.info(f"vLLM server process exists (PID {pid}) but not ready yet, waiting for startup...")
        return wait_for_server_ready(endpoint, max_wait_seconds=max_wait_seconds)
    
    # No server process - need to restart
    logger.info("No vLLM server process found, attempting to restart...")
    return restart_vllm_server(cache_dir, max_wait_seconds=max_wait_seconds)


def restart_vllm_server(cache_dir: Path | None = None, max_wait_seconds: int = 300) -> bool:
    """Restart vLLM server using cached configuration.
    
    Args:
        cache_dir: Cache directory path. If None, uses default from _get_cache_dir()
        max_wait_seconds: Maximum time to wait for server to become ready
        
    Returns:
        True if server restarted and became ready, False otherwise
    """
    import subprocess
    
    if cache_dir is None:
        cache_dir = _get_cache_dir()
    
    config_file = cache_dir / "vllm_serve_config.json"
    
    if not config_file.exists():
        logger.error(f"vLLM server configuration not found at {config_file}. Cannot restart.")
        return False
    
    # Load cached configuration
    config = load_vllm_config(cache_dir)
    if config is None:
        logger.error(f"vLLM server configuration not found or invalid. Cannot restart.")
        return False
    
    # Check if old server is still running and stop it
    old_pid = get_vllm_server_pid(cache_dir)
    if old_pid is not None:
        logger.info(f"Stopping existing vLLM server (PID {old_pid}) before restart...")
        stop_vllm_server(cache_dir)
        time.sleep(2)  # Brief wait after stopping
    
    # Build command and environment using shared helpers
    cmd = _build_vllm_command(config)
    env = _build_vllm_env(config)
    
    # Start server as background process (using log files, not terminal)
    process = _start_vllm_process(cmd, env, cache_dir, config, stream_logs_to_terminal=False)
    if process is None:
        return False
    
    # Wait for server to be ready
    endpoint = f"http://localhost:{port}/v1"
    return wait_for_server_ready(endpoint, max_wait_seconds=max_wait_seconds)


@dataclass
class Prompts:
    # Query generation for literature crawling
    query_system: str
    query_user: str
    # Paper evaluation for crawled documents
    eval_system: str
    eval_user: str
    # Query generation for paper-to-trial matching
    match_query_system: str
    match_query_user: str
    # Paper-trial match evaluation
    match_eval_system: str
    match_eval_user: str
    # Seed query generation for matching (using high-quality matches)
    match_seed_query_system: str
    match_seed_query_user: str
    
    @classmethod
    def load(cls, path: Path = PROMPTS_PATH) -> "Prompts":
        with open(path) as f:
            data = json.load(f)
        return cls(
            query_system=data["query_generation"]["system"],
            query_user=data["query_generation"]["user"],
            eval_system=data["paper_evaluation"]["system"],
            eval_user=data["paper_evaluation"]["user"],
            match_query_system=data["query_generation_for_matching"]["system"],
            match_query_user=data["query_generation_for_matching"]["user"],
            match_eval_system=data["match_evaluation"]["system"],
            match_eval_user=data["match_evaluation"]["user"],
            match_seed_query_system=data["seed_query_generation_for_matching"]["system"],
            match_seed_query_user=data["seed_query_generation_for_matching"]["user"],
        )


def _strip_markdown(text: str) -> str:
    """Extract JSON from LLM output, handling markdown fences and surrounding text."""
    import re
    
    # Try to find JSON in code fences first
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    
    # Try to find raw JSON object or array
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()
    
    return text.strip()


class Watchdog:
    """Watchdog that monitors vLLM process and restarts it on failure."""
    
    def __init__(
        self,
        vllm_process: subprocess.Popen,
        cmd: list[str],
        env: dict[str, str],
        cache_dir: Path,
        config_data: dict,
        stream_logs_to_terminal: bool = True,
    ):
        """
        Initialize watchdog.
        
        Args:
            vllm_process: vLLM subprocess to monitor
            cmd: Command to restart vLLM
            env: Environment variables for vLLM
            cache_dir: Cache directory
            config_data: Configuration data
            stream_logs_to_terminal: Whether to stream logs to terminal on restart
        """
        self.vllm_process = vllm_process
        self.cmd = cmd
        self.env = env
        self.cache_dir = cache_dir
        self.config_data = config_data
        self.stream_logs_to_terminal = stream_logs_to_terminal
        self._restart_lock = threading.Lock()
        self._is_restarting = False
    
    def restart_vllm(self) -> bool:
        """Restart vLLM process (thread-safe).
        
        Returns:
            True if restart succeeded, False otherwise
        """
        with self._restart_lock:
            if self._is_restarting:
                logger.debug("Restart already in progress, skipping")
                return False
            
            self._is_restarting = True
        
        try:
            # Kill current process
            if self.vllm_process.poll() is None:
                logger.info("Stopping current vLLM process...")
                try:
                    self.vllm_process.terminate()
                    # Wait up to 10 seconds for graceful shutdown
                    try:
                        self.vllm_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.warning("Process did not stop gracefully, killing...")
                        self.vllm_process.kill()
                        self.vllm_process.wait()
                except Exception as e:
                    logger.warning(f"Error stopping process: {e}")
                    try:
                        self.vllm_process.kill()
                    except Exception:
                        pass
            
            # Clean up PID file
            pid_file = self.cache_dir / "vllm.pid"
            if pid_file.exists():
                pid_file.unlink()
            
            # Start new process
            logger.info("Restarting vLLM process...")
            new_process = _start_vllm_process(
                self.cmd,
                self.env,
                self.cache_dir,
                self.config_data,
                stream_logs_to_terminal=self.stream_logs_to_terminal,
            )
            
            if new_process is None:
                logger.error("Failed to restart vLLM process")
                return False
            
            self.vllm_process = new_process
            logger.info(f"vLLM process restarted (PID: {new_process.pid})")
            
            # Wait for server to be ready
            port = self.config_data.get("port", 8000)
            endpoint = f"http://localhost:{port}/v1"
            if wait_for_server_ready(endpoint, max_wait_seconds=300):
                logger.info("vLLM server is ready after restart")
                return True
            else:
                logger.error("vLLM server did not become ready after restart")
                return False
                
        finally:
            with self._restart_lock:
                self._is_restarting = False


def run_vllm_supervised(
    cmd: list[str],
    env: dict[str, str],
    cache_dir: Path,
    port: int,
    wrapper_port: int,
    config_data: dict,
    stream_logs_to_terminal: bool = True,
    poll_interval: int = 60,
) -> None:
    """Run vLLM in supervised mode with wrapper API and watchdog.

    Args:
        cmd: Command to start vLLM
        env: Environment variables
        cache_dir: Cache directory
        port: vLLM port
        wrapper_port: Wrapper API port
        config_data: Configuration data
        stream_logs_to_terminal: Whether to stream vLLM logs to terminal
        poll_interval: Seconds between process checks (default: 60)
    """
    try:
        from recite.crawler.vllm_wrapper import VLLMWrapper
    except ImportError:
        raise ImportError(
            "vllm_wrapper is not included in the minimal RECITE distribution. "
            "This feature requires a local vLLM server setup."
        )

    # Start vLLM subprocess
    logger.info("Starting vLLM subprocess...")
    vllm_process = _start_vllm_process(
        cmd, env, cache_dir, config_data, stream_logs_to_terminal=stream_logs_to_terminal
    )
    if vllm_process is None:
        raise RuntimeError("Failed to start vLLM process")

    # Create watchdog
    watchdog = Watchdog(
        vllm_process, cmd, env, cache_dir, config_data, stream_logs_to_terminal
    )

    # Create wrapper API
    vllm_endpoint = f"http://localhost:{port}/v1"
    wrapper = VLLMWrapper(
        vllm_endpoint=vllm_endpoint,
        watchdog=watchdog,
        health_check_interval=10,
        failure_threshold=3,
    )
    
    # Start wrapper API in background thread
    logger.info(f"Starting wrapper API on port {wrapper_port}...")
    wrapper_thread = wrapper.start_server_thread(port=wrapper_port)
    
    # Save watchdog PID (this is the serve process PID)
    watchdog_pid_file = cache_dir / "serve_watchdog.pid"
    watchdog_pid_file.write_text(str(os.getpid()))
    logger.info(f"Watchdog PID saved to {watchdog_pid_file}")
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        # Kill vLLM child
        if vllm_process.poll() is None:
            try:
                vllm_process.terminate()
                vllm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vllm_process.kill()
        # Clean up PID files
        pid_file = cache_dir / "vllm.pid"
        if pid_file.exists():
            pid_file.unlink()
        if watchdog_pid_file.exists():
            watchdog_pid_file.unlink()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run watchdog loop
    logger.info("Watchdog loop started. Monitoring vLLM process...")
    try:
        while True:
            time.sleep(poll_interval)
            
            # Check if process has exited
            if vllm_process.poll() is not None:
                logger.warning(f"vLLM process exited with code {vllm_process.returncode}, restarting...")
                if not watchdog.restart_vllm():
                    logger.error("Failed to restart vLLM, exiting watchdog")
                    break
    except KeyboardInterrupt:
        logger.info("Watchdog loop interrupted")
    finally:
        # Cleanup
        if vllm_process.poll() is None:
            try:
                vllm_process.terminate()
                vllm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                vllm_process.kill()
        wrapper.close()
        pid_file = cache_dir / "vllm.pid"
        if pid_file.exists():
            pid_file.unlink()
        if watchdog_pid_file.exists():
            watchdog_pid_file.unlink()


def _build_fix_prompt(raw_output: str, extracted: str, error: str) -> str:
    """Build a prompt asking the LLM to fix its JSON output."""
    return f"""Your previous response could not be parsed as valid JSON.

Raw output:
{raw_output[:500]}{"..." if len(raw_output) > 500 else ""}

Extracted JSON attempt:
{extracted[:500]}{"..." if len(extracted) > 500 else ""}

Parse error: {error}

Please respond with ONLY valid JSON, no explanation or markdown. Ensure all strings are properly escaped (use \\" for quotes inside strings)."""


class LLMClient:
    """Simple vLLM client with JSON retry logic.
    
    By default, connects to wrapper API endpoint (http://localhost:8001/v1)
    which provides self-healing capabilities. The wrapper API handles
    restarts automatically, so revive_llm is no longer needed.
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8001/v1",
        model: str | None = None,
        max_retries: int = 2,
        revive_llm: bool = False,
        cache_dir: Path | None = None,
        wait_for_revive_seconds: int = 0,
    ):
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=120)
        self.prompts = Prompts.load()
        self.max_retries = max_retries
        self.revive_llm = revive_llm  # Kept for backward compatibility, but no longer used
        self.cache_dir = cache_dir
        self.wait_for_revive_seconds = wait_for_revive_seconds
        self.model = model or self._get_model()
    
    def _get_model(self) -> str:
        """Get the first available model from the server."""
        try:
            resp = self.client.get(f"{self.endpoint}/models", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if not models:
                raise RuntimeError("No models available on vLLM server")
            return models[0]["id"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to get models from vLLM server: {e.response.status_code} - {e.response.text[:200]}")
            raise RuntimeError(f"vLLM server error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            # Connection error - wrapper API should handle restarts, so just raise
            logger.error(f"Failed to connect to server at {self.endpoint}: {e}")
            raise RuntimeError(f"Cannot connect to server at {self.endpoint}") from e
    
    def _call_llm(self, messages: list[dict], max_retries: int = 3, base_delay: float = 1.0) -> str:
        """Make a raw LLM call with retry logic for transient errors."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                resp = self.client.post(
                    f"{self.endpoint}/chat/completions",
                    json={"model": self.model, "messages": messages, "temperature": 0},
                    timeout=120,
                )
                
                # Handle different status codes
                if resp.status_code == 200:
                    result = resp.json()
                    if "choices" not in result or not result["choices"]:
                        raise ValueError("Empty response from LLM")
                    return result["choices"][0]["message"]["content"]
                
                # 400 Bad Request - usually means invalid request format or model issue
                elif resp.status_code == 400:
                    error_text = resp.text[:500]
                    logger.error(f"Bad request (400) from vLLM server. Model: {self.model}, Error: {error_text}")
                    # Don't retry 400 errors - they're usually configuration issues
                    raise httpx.HTTPStatusError(
                        f"Bad request to vLLM server: {error_text}",
                        request=resp.request,
                        response=resp,
                    )
                
                # 429 Too Many Requests or 503 Service Unavailable - retry with backoff
                elif resp.status_code in (429, 503):
                    if attempt < max_retries - 1:
                        if self.wait_for_revive_seconds > 0 and wait_for_server_ready(
                            self.endpoint, max_wait_seconds=self.wait_for_revive_seconds
                        ):
                            continue
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Server busy ({resp.status_code}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        resp.raise_for_status()
                
                # Other 4xx/5xx errors
                else:
                    resp.raise_for_status()
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 400:
                    # Don't retry 400 errors
                    raise
                if attempt < max_retries - 1 and e.response.status_code >= 500:
                    if self.wait_for_revive_seconds > 0 and wait_for_server_ready(
                        self.endpoint, max_wait_seconds=self.wait_for_revive_seconds
                    ):
                        continue
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Server error ({e.response.status_code}), retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                raise
                
            except httpx.RequestError as e:
                last_error = e
                # Connection error - optionally wait for server (e.g. wrapper reviving) then retry
                if attempt < max_retries - 1:
                    if self.wait_for_revive_seconds > 0 and wait_for_server_ready(
                        self.endpoint, max_wait_seconds=self.wait_for_revive_seconds
                    ):
                        continue
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Request error, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay)
                    continue
                raise
        
        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError("Failed to call LLM after retries")
    
    def _try_parse_json(self, text: str) -> tuple[dict | None, str | None, str]:
        """Try to parse JSON from text. Returns (parsed, error, extracted)."""
        extracted = _strip_markdown(text)
        try:
            return json.loads(extracted), None, extracted
        except json.JSONDecodeError as e:
            return None, str(e), extracted
    
    def complete_json(self, prompt: str, system: str = "") -> dict | None:
        """Call LLM and parse JSON response, with retry on parse failure."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        # First attempt
        try:
            raw_output = self._call_llm(messages)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                logger.error(f"Bad request to LLM (400). This may indicate a model configuration issue. Model: {self.model}")
                logger.error(f"Error details: {e.response.text[:500]}")
            return None
        except (httpx.RequestError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to call LLM: {e}")
            return None
        
        parsed, error, extracted = self._try_parse_json(raw_output)
        
        if parsed is not None:
            return parsed
        
        # Retry loop with fix prompts
        for attempt in range(self.max_retries):
            fix_prompt = _build_fix_prompt(raw_output, extracted, error)
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": fix_prompt})
            
            try:
                raw_output = self._call_llm(messages)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400:
                    logger.error(f"Bad request during JSON fix retry (400). Model: {self.model}")
                return None
            except (httpx.RequestError, RuntimeError, ValueError) as e:
                logger.error(f"Failed to call LLM during retry: {e}")
                return None
            
            parsed, error, extracted = self._try_parse_json(raw_output)
            
            if parsed is not None:
                return parsed
        
        # All retries failed
        logger.warning(f"Failed to parse JSON after {self.max_retries} retries")
        return None
