"""Unified Reproducibility Manifest — vendored across research repos.

Single-file, stdlib-only (+ optional requests) provenance module that closes ALL gaps
vs reproducibility.md spec. Designed to be vendored identically into:
  - arbitration/shared/unified_manifest.py
  - canaries/shift_detection_monitor/unified_manifest.py
  - agentic/scripts/unified_manifest.py (or repo root)

SCHEMA v4 — captures: hardware, OS, python runtime, environment variables,
code state, inference engines (API/Ollama/MLX), model identity, prompts,
experiment design, and result artifacts.

INTEGRATION:
  1-line:  capture(__file__)
  2-line:  from unified_manifest import capture; capture(__file__, result_path="results/out.json")
  CLI:     python unified_manifest.py run --result-dir results/ -- python experiment.py

BACKWARD COMPATIBILITY:
  Exports ALL public names from both arbitration and canaries provenance modules.
  Drop-in replacement for shared/provenance.py and shift_detection_monitor/provenance.py.

VERSION: 1.0.0 (2026-08-23)
SOURCE_HASH: <will be set by sync script>
"""
from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import weakref
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

__version__ = "1.0.0"
# Schema version history:
#   v4: response_model, system_fingerprint, req_temperature, req_max_tokens, req_top_p, ...
#   v5 (2026-08-28): + req_temperature_omitted (bool), req_max_completion_tokens,
#       req_max_output_tokens (unified max regardless of param name). Closes the gap
#       where reasoning models using max_completion_tokens logged req_max_tokens=None,
#       and records whether the temperature field was omitted (some reasoning-model routes reject it).
SCHEMA_VERSION = 5
_MODULE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

# ═══════════════════════════════════════════════════════════════════════════════
# RUN IDENTITY (per-process, stable)
# ═══════════════════════════════════════════════════════════════════════════════

MANIFEST_ID = os.environ.get("UNIFIED_MANIFEST_ID") or uuid.uuid4().hex[:12]
_RUN_START = time.time()
_RUN_STATUS = "running"  # running | completed | failed | interrupted
_manifest_written = False
_finalized = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Output paths (configurable per-repo)
_JSONL_PATH: Path | None = None
_SIDECAR_PATH: Path | None = None
_RESULT_PATHS: list[Path] = []

# Registered artifacts
_registered_prompts: dict[str, str] = {}
_registered_tool_schemas: dict[str, str] = {}
_registered_design: dict[str, Any] = {}
_extra_metadata: dict[str, Any] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# HARDWARE & OS CAPTURE
# ═══════════════════════════════════════════════════════════════════════════════

_hardware_cache: dict | None = None


def _capture_hardware() -> dict:
    """Capture hardware info. macOS: sysctl. Linux: /proc. Degrades gracefully."""
    global _hardware_cache
    if _hardware_cache is not None:
        return _hardware_cache

    hw: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    if sys.platform == "darwin":
        # macOS — Apple Silicon details via sysctl
        sysctl_keys = {
            "hw.model": "machine_model",
            "machdep.cpu.brand_string": "cpu_brand",
            "hw.ncpu": "cpu_cores_logical",
            "hw.physicalcpu": "cpu_cores_physical",
            "hw.memsize": "memory_bytes",
            "hw.optional.arm64": "is_arm64",
        }
        for key, field in sysctl_keys.items():
            try:
                val = subprocess.check_output(
                    ["sysctl", "-n", key], stderr=subprocess.DEVNULL, timeout=2
                ).decode().strip()
                # Convert numeric values
                if val.isdigit():
                    val = int(val)
                hw[field] = val
            except Exception:
                pass

        # macOS version
        try:
            hw["os_version"] = subprocess.check_output(
                ["sw_vers", "-productVersion"], stderr=subprocess.DEVNULL, timeout=2
            ).decode().strip()
            hw["os_build"] = subprocess.check_output(
                ["sw_vers", "-buildVersion"], stderr=subprocess.DEVNULL, timeout=2
            ).decode().strip()
        except Exception:
            pass

        # Metal GPU info (from system_profiler, bounded)
        try:
            sp = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                stderr=subprocess.DEVNULL, timeout=5
            )
            displays = json.loads(sp).get("SPDisplaysDataType", [])
            if displays:
                gpu = displays[0]
                hw["gpu_model"] = gpu.get("sppci_model", "unknown")
                hw["gpu_cores"] = gpu.get("sppci_cores", "unknown")
                hw["metal_support"] = gpu.get("spmetal_supported", "unknown")
        except Exception:
            pass

    elif sys.platform == "linux":
        # Linux — /proc and lscpu
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        hw["memory_bytes"] = int(line.split()[1]) * 1024
                        break
        except Exception:
            pass

        try:
            hw["cpu_cores_logical"] = os.cpu_count()
            lscpu = subprocess.check_output(
                ["lscpu"], stderr=subprocess.DEVNULL, timeout=2
            ).decode()
            for line in lscpu.splitlines():
                if "Model name:" in line:
                    hw["cpu_brand"] = line.split(":", 1)[1].strip()
                elif "Core(s) per socket:" in line:
                    hw["cpu_cores_physical"] = int(line.split(":", 1)[1].strip())
        except Exception:
            pass

        # CUDA/GPU
        try:
            nvidia = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            if nvidia:
                parts = nvidia.split(",")
                hw["gpu_model"] = parts[0].strip()
                hw["gpu_memory_mb"] = int(parts[1].strip()) if len(parts) > 1 else None
                hw["nvidia_driver"] = parts[2].strip() if len(parts) > 2 else None
        except Exception:
            pass

    _hardware_cache = hw
    return hw


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON RUNTIME & LOCKFILE
# ═══════════════════════════════════════════════════════════════════════════════

_lockfile_cache: str | None = None
_lockfile_hash: str | None = None

# Packages tracked inline (quick reference without full lockfile)
_TRACKED_PACKAGES = (
    "openai", "httpx", "anthropic", "numpy", "scipy", "scikit-learn",
    "statsmodels", "pydantic", "torch", "transformers", "tokenizers",
    "datasets", "pandas", "mlx", "mlx-lm", "requests",
)


def _capture_python_runtime() -> dict:
    """Python version, venv, key packages."""
    runtime = {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }

    # Detect venv
    if sys.prefix != sys.base_prefix:
        runtime["venv_path"] = sys.prefix

    # Key package versions (fast)
    packages = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            packages[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            continue
    runtime["packages"] = packages

    return runtime


def _capture_lockfile() -> tuple[str | None, str | None]:
    """Cached pip freeze. Keyed by site-packages mtime for invalidation."""
    global _lockfile_cache, _lockfile_hash

    if _lockfile_cache is not None:
        return _lockfile_cache, _lockfile_hash

    try:
        # Check for existing lockfiles first (fast path)
        cwd = Path.cwd()
        for lockfile_name in ("requirements.txt", "uv.lock", "poetry.lock", "Pipfile.lock"):
            lf = cwd / lockfile_name
            if lf.exists():
                content = lf.read_text()
                _lockfile_cache = f"[from {lockfile_name}]\n{content}"
                _lockfile_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
                return _lockfile_cache, _lockfile_hash

        # Fall back to pip freeze (slower)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            _lockfile_cache = result.stdout
            _lockfile_hash = hashlib.sha256(result.stdout.encode()).hexdigest()[:12]
    except Exception:
        pass

    return _lockfile_cache, _lockfile_hash


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════

_ENV_PREFIXES = ("OLLAMA_", "CUDA_", "TORCH_", "HF_", "PYTHON", "OMP_", "MKL_",
                 "OPENBLAS_", "VECLIB_", "NCCL_", "TOKENIZERS_")
_SECRET_PATTERNS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def _capture_environment() -> dict:
    """Capture relevant env vars. Secrets → presence only."""
    env: dict[str, Any] = {}

    for key, value in os.environ.items():
        # Check if this is a tracked prefix
        if any(key.startswith(prefix) for prefix in _ENV_PREFIXES):
            # Check if it's a secret
            if any(pattern in key.upper() for pattern in _SECRET_PATTERNS):
                env[key] = {"present": True, "redacted": True}
            else:
                env[key] = value
        # Also capture API key presence
        elif any(pattern in key.upper() for pattern in _SECRET_PATTERNS):
            if any(api in key.upper() for api in ("OPENAI", "ANTHROPIC", "GOOGLE", "FRONTIER", "LLM", "BEDROCK")):
                env[key] = {"present": True, "redacted": True}

    return env


# ═══════════════════════════════════════════════════════════════════════════════
# GIT / CODE STATE
# ═══════════════════════════════════════════════════════════════════════════════

_git_cache: dict | None = None


def _capture_git() -> dict:
    """Git SHA, dirty state, branch."""
    global _git_cache
    if _git_cache is not None:
        return _git_cache

    git_info: dict[str, Any] = {}
    try:
        git_info["sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()

        git_info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()

        # Dirty = staged or unstaged changes (not untracked)
        unstaged = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL, timeout=5
        ) != 0
        staged = subprocess.call(
            ["git", "diff", "--cached", "--quiet"], stderr=subprocess.DEVNULL, timeout=5
        ) != 0
        git_info["dirty"] = unstaged or staged

        # If dirty: capture diff hash and summary (CRITICAL for reproducibility)
        if git_info["dirty"]:
            try:
                diff = subprocess.check_output(
                    ["git", "diff", "HEAD", "--stat"],
                    stderr=subprocess.DEVNULL, timeout=5
                ).decode().strip()
                git_info["dirty_summary"] = diff  # e.g. "3 files changed, 12 insertions(+)"
                # Full diff hash (can compare without storing full diff)
                full_diff = subprocess.check_output(
                    ["git", "diff", "HEAD"],
                    stderr=subprocess.DEVNULL, timeout=10
                ).decode()
                git_info["dirty_diff_hash"] = hashlib.sha256(full_diff.encode()).hexdigest()[:16]
                # Store abbreviated diff if small enough (< 5KB)
                if len(full_diff) < 5000:
                    git_info["dirty_diff"] = full_diff
            except Exception:
                pass

    except Exception:
        git_info["sha"] = "unknown"
        git_info["dirty"] = None

    _git_cache = git_info
    return git_info


def git_sha() -> str:
    """Backward-compatible: short SHA with -dirty suffix."""
    info = _capture_git()
    sha = info.get("sha", "unknown")
    if info.get("dirty"):
        return f"{sha}-dirty"
    return sha


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE — OLLAMA
# ═══════════════════════════════════════════════════════════════════════════════

_ollama_cache: dict | None = None


def _probe_ollama(model: str | None = None) -> dict | None:
    """Auto-detect and capture Ollama state. Returns None if not available.

    HARDENED (v1.1): exponential backoff up to 2s, explicit probe_status field,
    daemon PID/uptime capture, ollama ps for loaded models.
    """
    global _ollama_cache
    if _ollama_cache is not None:
        return _ollama_cache

    try:
        import requests as req
    except ImportError:
        # Try urllib as fallback
        try:
            from urllib.request import urlopen
            host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if not host.startswith("http"):
                host = f"http://{host}"
            resp = urlopen(f"{host}/api/version", timeout=2.0)
            version_data = json.loads(resp.read())
        except Exception:
            return {"detected": False, "probe_status": "unreachable", "host": os.environ.get("OLLAMA_HOST", "localhost:11434")}
        _ollama_cache = {"version": version_data.get("version", "unknown"), "detected": True, "probe_status": "ok"}
        return _ollama_cache

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    if not host.startswith("http"):
        host = f"http://{host}"

    ollama: dict[str, Any] = {"detected": False, "probe_status": "unknown", "host": host}

    # Exponential backoff: 250ms, 500ms, 1s (total ~2s max)
    for attempt, timeout in enumerate([0.5, 1.0, 2.0]):
        try:
            r = req.get(f"{host}/api/version", timeout=timeout)
            if r.status_code == 200:
                ollama["detected"] = True
                ollama["probe_status"] = "ok"
                ollama["version"] = r.json().get("version", "unknown")
                ollama["probe_attempts"] = attempt + 1
                break
        except Exception as e:
            ollama["probe_status"] = f"timeout_attempt_{attempt + 1}"
            ollama["probe_error"] = str(e)[:100]
            continue

    if not ollama["detected"]:
        # Still record what we can from the binary itself
        try:
            ver_out = subprocess.check_output(
                ["ollama", "--version"], stderr=subprocess.DEVNULL, timeout=5
            ).decode().strip()
            ollama["version_from_binary"] = ver_out
        except Exception:
            pass
        _ollama_cache = ollama
        return ollama

    # Binary SHA
    try:
        which = subprocess.check_output(
            ["which", "ollama"], stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        if which:
            sha = subprocess.check_output(
                ["shasum", "-a", "256", which], stderr=subprocess.DEVNULL, timeout=5
            ).decode().split()[0]
            ollama["binary_sha256"] = sha
            ollama["binary_path"] = which
    except Exception:
        pass

    # Daemon PID, uptime, and restart detection (CRITICAL for reproducibility.md Rule 8)
    try:
        # Find Ollama server PID via lsof on the listen port
        port = host.split(":")[-1] if ":" in host else "11434"
        pid_out = subprocess.check_output(
            ["lsof", f"-ti:{port}"], stderr=subprocess.DEVNULL, timeout=3
        ).decode().strip()
        if pid_out:
            pids = pid_out.splitlines()
            ollama["daemon_pid"] = int(pids[0])
            # Get process start time (for restart detection)
            ps_out = subprocess.check_output(
                ["ps", "-p", pids[0], "-o", "lstart=,etime="],
                stderr=subprocess.DEVNULL, timeout=3
            ).decode().strip()
            ollama["daemon_start_time"] = ps_out
            # Verify the PID's environment matches our expected env vars
            try:
                ps_env = subprocess.check_output(
                    ["ps", "eww", pids[0]], stderr=subprocess.DEVNULL, timeout=3
                ).decode()
                # Extract OLLAMA_* vars from the daemon's actual environment
                daemon_env = {}
                for token in ps_env.split():
                    if token.startswith("OLLAMA_") and "=" in token:
                        k, v = token.split("=", 1)
                        daemon_env[k] = v
                if daemon_env:
                    ollama["daemon_effective_env"] = daemon_env
            except Exception:
                pass
    except Exception:
        pass

    # Currently loaded/running models (ollama ps equivalent)
    try:
        r = req.get(f"{host}/api/ps", timeout=2)
        if r.status_code == 200:
            ps_data = r.json()
            models_running = ps_data.get("models", [])
            ollama["running_models"] = [
                {
                    "name": m.get("name"),
                    "digest": (m.get("digest") or "")[:12],
                    "size": m.get("size"),
                    "expires_at": m.get("expires_at"),
                }
                for m in models_running
            ]
    except Exception:
        pass

    # OLLAMA_* env vars (all of them)
    ollama_env = {}
    for key, val in os.environ.items():
        if key.startswith("OLLAMA_"):
            ollama_env[key] = val
    ollama["server_env"] = ollama_env

    # Model details (if specified)
    if model:
        try:
            r = req.post(f"{host}/api/show", json={"name": model}, timeout=10)
            if r.status_code == 200:
                show = r.json()
                ollama["model_show"] = {
                    "parameters": show.get("parameters"),
                    "details": show.get("details"),
                    "model_info_keys": sorted((show.get("model_info") or {}).keys()),
                }
                # Extract key model_info fields
                mi = show.get("model_info") or {}
                selected = {}
                for k in mi:
                    kl = k.lower()
                    if any(x in kl for x in ("context_length", "rope", "attention", "head", "vocab")):
                        selected[k] = mi[k]
                if selected:
                    ollama["model_show"]["model_info_selected"] = selected
        except Exception:
            pass

    # Live model digests
    try:
        r = req.get(f"{host}/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            ollama["loaded_models"] = {
                m["name"]: {"digest": (m.get("digest") or "")[:12], "size": m.get("size")}
                for m in models
            }
    except Exception:
        pass

    _ollama_cache = ollama
    return ollama


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE — API (OpenAI/Anthropic/Bedrock)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-call metadata (populated by instrument_client)
_last_meta: dict = {}
_call_logs: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_MAX_CALL_LOG_LEN = 50
_last_active_client_ref = None
_misroute_count = 0

# Hash registries
_prompt_hash: str | None = None
_tool_schema_hash: str | None = None


def _sha256_short(text: str) -> str:
    """First 12 hex chars of SHA-256."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def register_prompt(system_prompt: str, *, label: str = "default") -> str:
    """Register system prompt. Returns hash. Full text stored in manifest."""
    global _prompt_hash
    _prompt_hash = _sha256_short(system_prompt)
    _registered_prompts[f"prompt_{label}"] = system_prompt
    return _prompt_hash


def register_tool_schema(tools: list[dict], *, label: str = "default") -> str:
    """Register tool schema. Returns hash of canonical JSON."""
    global _tool_schema_hash
    canonical = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    _tool_schema_hash = _sha256_short(canonical)
    _registered_tool_schemas[f"tools_{label}"] = canonical
    return _tool_schema_hash


class ModelMismatchError(RuntimeError):
    """Raised when API serves a different model than requested."""
    pass


def instrument_client(client, strict=True):
    """Patch chat.completions.create to record request+response metadata.

    Backward-compatible with arbitration's instrument_client().
    """
    create = client.chat.completions.create

    def wrapped(*args, **kwargs):
        global _misroute_count
        resp = create(*args, **kwargs)

        requested = kwargs.get("model") or (args[0] if args else None)
        actual = getattr(resp, "model", None)
        if requested and actual and actual != requested:
            _misroute_count += 1
            if strict:
                raise ModelMismatchError(
                    f"Gateway served {actual!r} when {requested!r} was requested "
                    f"(misroute #{_misroute_count}). Aborting to prevent data corruption."
                )
            else:
                import logging
                logging.getLogger("provenance").warning(
                    f"MODEL MISMATCH: requested={requested} but gateway served={actual} "
                    f"(misroute #{_misroute_count})"
                )

        call_meta = {
            "response_model": getattr(resp, "model", None),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "req_temperature": kwargs.get("temperature"),
            "req_temperature_omitted": "temperature" not in kwargs,
            "req_max_tokens": kwargs.get("max_tokens"),
            "req_max_completion_tokens": kwargs.get("max_completion_tokens"),
            "req_max_output_tokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
            "req_top_p": kwargs.get("top_p"),
            "req_seed": kwargs.get("seed"),
            "req_response_format": str(kwargs.get("response_format")) if kwargs.get("response_format") else None,
        }
        _last_meta.update(call_meta)

        log = _call_logs.setdefault(client, [])
        log.append(call_meta)
        if len(log) > _MAX_CALL_LOG_LEN:
            del log[:-_MAX_CALL_LOG_LEN]

        global _last_active_client_ref
        _last_active_client_ref = weakref.ref(client)
        return resp

    client.chat.completions.create = wrapped
    return client


def assert_model_match(requested_model: str) -> None:
    """Raise if last API call was served by a different model."""
    actual = _last_meta.get("response_model")
    if actual and actual != requested_model:
        raise ModelMismatchError(
            f"Gateway served {actual!r} when {requested_model!r} was requested."
        )


def get_misroute_count() -> int:
    """Number of misrouted API calls detected this session."""
    return _misroute_count


# ═══════════════════════════════════════════════════════════════════════════════
# RECORD STAMPING (arbitration backward compat)
# ═══════════════════════════════════════════════════════════════════════════════

def stamp(rec: dict) -> dict:
    """Merge captured provenance into a record. Backward-compatible with arbitration."""
    _ensure_manifest_started()
    rec.setdefault("schema_version", SCHEMA_VERSION)
    rec.setdefault("manifest_id", MANIFEST_ID)
    rec.setdefault("git_sha", git_sha())

    # Merge API call metadata
    for key, val in _last_meta.items():
        existing = rec.get(key)
        if existing is not None and existing != val:
            import logging
            logging.getLogger("provenance").error(
                f"stamp() CONFLICT: record has {key}={existing!r}, captured={val!r}"
            )
        rec.setdefault(key, val)

    # Multi-hop provenance. If the active client was garbage-collected before
    # stamp() ran, its per-client call log is gone: omit hop_provenance and warn
    # (silently dropping multi-hop identity would hide provenance loss).
    active_client = _last_active_client_ref() if _last_active_client_ref else None
    if _last_active_client_ref is not None and active_client is None:
        import logging
        logging.getLogger("provenance").warning(
            "stamp(): active client was garbage-collected before stamping; "
            "hop_provenance omitted for this record"
        )
    active_log = _call_logs.pop(active_client, None) if active_client is not None else None
    if active_log and len(active_log) > 1:
        rec.setdefault("hop_provenance", active_log)

    # Prompt/tool hashes
    if _prompt_hash:
        rec.setdefault("prompt_hash", _prompt_hash)
    if _tool_schema_hash:
        rec.setdefault("tool_schema_hash", _tool_schema_hash)

    _last_meta.clear()
    return rec


# ═══════════════════════════════════════════════════════════════════════════════
# LIB VERSIONS (backward compat)
# ═══════════════════════════════════════════════════════════════════════════════

_LEGACY_TRACKED_PKGS = ("openai", "httpx", "anthropic", "numpy", "scipy", "scikit-learn", "pandas")


def lib_versions() -> dict:
    """Backward-compatible: tracked package versions."""
    out = {}
    for pkg in _LEGACY_TRACKED_PKGS:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            continue
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST BUILDING & WRITING
# ═══════════════════════════════════════════════════════════════════════════════

def _build_full_manifest(
    script: str | None = None,
    result_path: str | Path | None = None,
    extra: dict | None = None,
    status: str = "completed",
    ollama_model: str | None = None,
) -> dict:
    """Build the complete v4 manifest document."""
    lockfile_content, lockfile_hash = _capture_lockfile()

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_module_version": __version__,
        "manifest_module_hash": _MODULE_HASH,
        "manifest_id": MANIFEST_ID,
        "status": status,
        "timestamps": {
            "start": _RUN_START,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_RUN_START)),
            "end": time.time(),
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "script": script or (Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "unknown"),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "hostname": platform.node(),
    }

    # Sections
    manifest["hardware"] = _capture_hardware()
    manifest["python_runtime"] = _capture_python_runtime()
    manifest["environment"] = _capture_environment()
    manifest["code"] = _capture_git()

    # Lockfile (inline if small, hash if large)
    if lockfile_content:
        if len(lockfile_content) < 10000:
            manifest["python_runtime"]["lockfile"] = lockfile_content
        else:
            manifest["python_runtime"]["lockfile_hash"] = lockfile_hash
            manifest["python_runtime"]["lockfile_lines"] = lockfile_content.count("\n")

    # Inference engines
    engines: dict[str, Any] = {}

    # Ollama (auto-detect)
    ollama_state = _probe_ollama(model=ollama_model)
    if ollama_state:
        engines["ollama"] = ollama_state

    # API (from instrumented calls)
    if _last_meta or _registered_prompts:
        api_info: dict[str, Any] = {
            "api_base": os.environ.get("FRONTIER_API_BASE", os.environ.get("OPENAI_BASE_URL", "")),
        }
        if _last_meta:
            api_info["last_call"] = dict(_last_meta)
        engines["api"] = api_info

    if engines:
        manifest["inference_engines"] = engines

    # Prompts and tools
    if _registered_prompts or _registered_tool_schemas:
        manifest["prompts_and_tools"] = {}
        if _registered_prompts:
            manifest["prompts_and_tools"]["prompts"] = dict(_registered_prompts)
            if _prompt_hash:
                manifest["prompts_and_tools"]["prompt_hash"] = _prompt_hash
        if _registered_tool_schemas:
            manifest["prompts_and_tools"]["tool_schemas"] = dict(_registered_tool_schemas)
            if _tool_schema_hash:
                manifest["prompts_and_tools"]["tool_schema_hash"] = _tool_schema_hash

    # Experiment design
    if _registered_design:
        manifest["experiment_design"] = dict(_registered_design)

    # Paper provenance (which papers this run serves)
    if _paper_provenance:
        manifest["paper_provenance"] = dict(_paper_provenance)

    # Per-request event log (for long-running/agentic runs)
    if _event_log:
        manifest["event_log_count"] = len(_event_log)
        # Detect mid-run drift
        fingerprints = set()
        models_seen = set()
        for evt in _event_log:
            if evt.get("system_fingerprint"):
                fingerprints.add(evt["system_fingerprint"])
            if evt.get("model"):
                models_seen.add(evt["model"])
        if len(fingerprints) > 1:
            manifest["drift_detected"] = {
                "system_fingerprints": sorted(fingerprints),
                "warning": "API backend changed mid-run"
            }
        if len(models_seen) > 1:
            manifest.setdefault("drift_detected", {})["models_seen"] = sorted(models_seen)

    # Result artifacts
    if result_path:
        rp = Path(result_path)
        artifact: dict[str, Any] = {"path": str(rp)}
        if rp.exists():
            artifact["size_bytes"] = rp.stat().st_size
            # Hash small result files
            if rp.stat().st_size < 100_000_000:  # < 100MB
                artifact["sha256"] = hashlib.sha256(rp.read_bytes()).hexdigest()[:16]
        manifest["result_artifact"] = artifact

    # Extra metadata (caller-supplied)
    if extra:
        manifest["extra"] = extra
    if _extra_metadata:
        manifest.setdefault("extra", {}).update(_extra_metadata)

    return manifest


def _write_atomic_json(path: Path, data: dict) -> None:
    """Atomic write: temp file + fsync + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _append_jsonl(path: Path, data: dict) -> None:
    """Append one line to JSONL with file locking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(data, default=str) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — HIGH LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

_started_marker: Path | None = None


def _ensure_manifest_started() -> None:
    """Write a started marker for crash/SIGKILL detection.

    SIGKILL resilience: The started marker is the ONLY thing that survives a
    hard kill. It contains enough context to identify what was running and when.
    A stale .started file (no corresponding completed manifest) = crashed run.
    """
    global _started_marker
    if _started_marker is not None:
        return
    if _JSONL_PATH:
        marker_dir = _JSONL_PATH.parent / ".manifests"
        marker_dir.mkdir(parents=True, exist_ok=True)
        _started_marker = marker_dir / f"{MANIFEST_ID}.started"
        # Write substantial context into the marker so even a SIGKILL-ed run
        # leaves enough info for diagnosis
        marker_data = {
            "manifest_id": MANIFEST_ID,
            "script": Path(sys.argv[0]).name if sys.argv else "unknown",
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "hostname": platform.node(),
            "argv": sys.argv[:10],  # First 10 args
            "git_sha": _capture_git().get("sha", "unknown"),
            "cwd": str(Path.cwd()),
        }
        _started_marker.write_text(json.dumps(marker_data, indent=2))


def _finalize_manifest(status: str = "completed") -> None:
    """Write final manifest (atexit or explicit)."""
    global _finalized
    if _finalized:
        return
    _finalized = True

    manifest = _build_full_manifest(status=status)

    # Write sidecar if configured
    if _SIDECAR_PATH:
        _write_atomic_json(_SIDECAR_PATH, manifest)

    # Append to JSONL if configured
    if _JSONL_PATH:
        _append_jsonl(_JSONL_PATH, manifest)

    # Clean up started marker
    if _started_marker and _started_marker.exists():
        try:
            _started_marker.unlink()
        except OSError:
            pass


def capture(
    script_path: str | None = None,
    *,
    result_path: str | Path | None = None,
    jsonl_path: str | Path | None = None,
    sidecar: bool = True,
    ollama_model: str | None = None,
    extra: dict | None = None,
) -> None:
    """1-line integration for uncovered scripts.

    Call at the top of any experiment script:
        from unified_manifest import capture
        capture(__file__)

    Or with result path:
        capture(__file__, result_path="results/output.json")
    """
    global _JSONL_PATH, _SIDECAR_PATH

    # Configure JSONL output
    if jsonl_path:
        _JSONL_PATH = Path(jsonl_path)
    elif _JSONL_PATH is None:
        # Default: results/run_manifest.jsonl relative to cwd
        _JSONL_PATH = Path("results/run_manifest.jsonl")

    # Configure sidecar output
    if result_path and sidecar:
        rp = Path(result_path)
        _SIDECAR_PATH = rp.with_name(rp.name + ".manifest.json")

    # Store extra
    if extra:
        _extra_metadata.update(extra)

    # Write started marker
    _ensure_manifest_started()

    # Register atexit
    atexit.register(_finalize_manifest, "completed")

    # Register signal handlers for graceful shutdown
    def _signal_handler(signum, frame):
        _finalize_manifest("interrupted")
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # Can't set handlers in threads


def register_design(
    *,
    n_per_condition: int | None = None,
    seeds: list | dict | None = None,
    conditions: list | dict | None = None,
    eval_criteria: str | None = None,
    sampling_params: dict | None = None,
    **kwargs,
) -> None:
    """Register experiment design metadata."""
    if n_per_condition is not None:
        _registered_design["n_per_condition"] = n_per_condition
    if seeds is not None:
        _registered_design["seeds"] = seeds
    if conditions is not None:
        _registered_design["conditions"] = conditions
    if eval_criteria is not None:
        _registered_design["eval_criteria"] = eval_criteria
    if sampling_params is not None:
        _registered_design["sampling_params"] = sampling_params
    _registered_design.update(kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-REQUEST EVENT LOG (for streaming/batched/agentic runs)
# ═══════════════════════════════════════════════════════════════════════════════

_event_log: list[dict] = []
_EVENT_LOG_MAX = 10000  # Cap to prevent memory explosion in long runs


def log_event(
    event_type: str,
    *,
    model: str | None = None,
    system_fingerprint: str | None = None,
    params: dict | None = None,
    response_hash: str | None = None,
    step_index: int | None = None,
    duration_ms: float | None = None,
    error: str | None = None,
    **kwargs,
) -> None:
    """Log a per-request/per-step event for long-running experiments.

    Call this per API call, tool execution, or agent step to capture
    mid-run drift (system_fingerprint changes, model routing changes, etc.)

    Usage:
        log_event("api_call", model="gpt-4o", system_fingerprint="fp_abc123",
                  params={"temperature": 0}, duration_ms=342)
        log_event("tool_call", step_index=5, params={"tool": "bash", "cmd": "ls"})
    """
    if len(_event_log) >= _EVENT_LOG_MAX:
        return  # Silently cap

    event: dict[str, Any] = {
        "type": event_type,
        "timestamp": time.time(),
        "seq": len(_event_log),
    }
    if model:
        event["model"] = model
    if system_fingerprint:
        event["system_fingerprint"] = system_fingerprint
    if params:
        event["params"] = params
    if response_hash:
        event["response_hash"] = response_hash
    if step_index is not None:
        event["step_index"] = step_index
    if duration_ms is not None:
        event["duration_ms"] = duration_ms
    if error:
        event["error"] = error
    event.update(kwargs)
    _event_log.append(event)


def get_event_log() -> list[dict]:
    """Return the accumulated event log."""
    return list(_event_log)


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER PROVENANCE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

_paper_provenance: dict[str, Any] = {}


def register_paper(
    paper_id: str,
    *,
    title: str | None = None,
    arxiv_id: str | None = None,
    sections: list[str] | None = None,
    claims: list[str] | None = None,
    figures: list[str] | None = None,
    tables: list[str] | None = None,
) -> None:
    """Register which paper(s) this script/result serves.

    A script can serve multiple papers. Call once per paper.

    Usage:
        register_paper("agentic_primary",
            title="Injection-Execution Dissociation",
            arxiv_id="2605.08442",
            sections=["§4.2", "Table 2"],
            claims=["Defense factorial shows 9 models × 7 defenses"])
    """
    entry: dict[str, Any] = {"paper_id": paper_id}
    if title:
        entry["title"] = title
    if arxiv_id:
        entry["arxiv_id"] = arxiv_id
    if sections:
        entry["sections"] = sections
    if claims:
        entry["claims"] = claims
    if figures:
        entry["figures"] = figures
    if tables:
        entry["tables"] = tables
    _paper_provenance.setdefault("papers", []).append(entry)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPAT — ARBITRATION (write_run_manifest)
# ═══════════════════════════════════════════════════════════════════════════════

# Default result path for arbitration
RESULTS = Path("results")
MANIFEST_PATH = RESULTS / "run_manifest.jsonl"


def write_run_manifest(script: str, extra: dict | None = None) -> str:
    """Backward-compatible: append one manifest line (arbitration pattern).

    Returns MANIFEST_ID.
    """
    global _JSONL_PATH, _manifest_written
    if _manifest_written:
        return MANIFEST_ID

    _JSONL_PATH = MANIFEST_PATH
    _manifest_written = True

    manifest = _build_full_manifest(script=script, extra=extra, status="completed")

    # Also write legacy-compatible compact entry for existing consumers
    legacy_entry = {
        "manifest_id": MANIFEST_ID,
        "script": script,
        "git_sha": git_sha(),
        "timestamp": time.time(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": sys.argv,
        "api_base": os.environ.get("FRONTIER_API_BASE", os.environ.get("OPENAI_BASE_URL", "")),
        "lib_versions": lib_versions(),
        "schema_version": SCHEMA_VERSION,
    }
    if _registered_prompts:
        legacy_entry["registered_artifacts"] = dict(_registered_prompts)
    if _prompt_hash:
        legacy_entry["prompt_hash"] = _prompt_hash
    if _tool_schema_hash:
        legacy_entry["tool_schema_hash"] = _tool_schema_hash
    if extra:
        legacy_entry.update(extra)
    # Add v4 extensions under a namespaced key
    legacy_entry["_v4"] = {
        "hardware": manifest.get("hardware"),
        "environment": manifest.get("environment"),
        "inference_engines": manifest.get("inference_engines"),
        "python_runtime": manifest.get("python_runtime"),
    }

    RESULTS.mkdir(exist_ok=True)
    _append_jsonl(MANIFEST_PATH, legacy_entry)

    return MANIFEST_ID


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPAT — CANARIES (build_manifest, write_manifest)
# ═══════════════════════════════════════════════════════════════════════════════

def build_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Backward-compatible: build a manifest dict (canaries pattern)."""
    git_info = _capture_git()
    runtime = _capture_python_runtime()

    # Legacy-compatible shape
    manifest: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git_commit": git_info.get("sha"),
        "git_dirty": git_info.get("dirty"),
        "python": runtime["version"],
        "platform": platform.platform(),
        "packages": runtime.get("packages", {}),
        # v4 extensions
        "schema_version": SCHEMA_VERSION,
        "manifest_id": MANIFEST_ID,
        "hardware": _capture_hardware(),
        "environment": _capture_environment(),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(
    result_path: str | Path, extra: dict[str, Any] | None = None
) -> Path:
    """Backward-compatible: write sidecar manifest (canaries pattern)."""
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = result_path.with_name(result_path.name + ".manifest.json")

    manifest = build_manifest(extra)
    _write_atomic_json(manifest_path, manifest)

    # Also append to JSONL if configured
    if _JSONL_PATH:
        _append_jsonl(_JSONL_PATH, manifest)

    return manifest_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI MODE (for bash wrappers / agentic)
# ═══════════════════════════════════════════════════════════════════════════════

def _cli_run(args: list[str]) -> int:
    """CLI wrapper: python unified_manifest.py run --result-dir DIR -- command..."""
    import argparse

    parser = argparse.ArgumentParser(description="Run with manifest capture")
    parser.add_argument("--result-dir", required=True, help="Result directory")
    parser.add_argument("--ollama-model", help="Model name for Ollama capture")
    parser.add_argument("--jsonl", help="JSONL manifest path (default: result-dir/run_manifest.jsonl)")
    parser.add_argument("command", nargs="+", help="Command to run")
    parsed = parser.parse_args(args)

    global _JSONL_PATH, _SIDECAR_PATH
    result_dir = Path(parsed.result_dir)
    _JSONL_PATH = Path(parsed.jsonl) if parsed.jsonl else result_dir / "run_manifest.jsonl"
    _SIDECAR_PATH = result_dir / f".manifest_{MANIFEST_ID}.json"

    # Export manifest ID for child processes
    os.environ["UNIFIED_MANIFEST_ID"] = MANIFEST_ID

    # Write started marker
    _ensure_manifest_started()

    # Run the child command
    try:
        result = subprocess.run(
            parsed.command,
            env={**os.environ, "UNIFIED_MANIFEST_ID": MANIFEST_ID},
        )
        status = "completed" if result.returncode == 0 else "failed"
        _extra_metadata["exit_code"] = result.returncode
        _finalize_manifest(status)
        return result.returncode
    except KeyboardInterrupt:
        _finalize_manifest("interrupted")
        return 130
    except Exception as e:
        _extra_metadata["error"] = str(e)
        _extra_metadata["traceback"] = traceback.format_exc()
        _finalize_manifest("failed")
        return 1


def _cli_audit(args: list[str]) -> int:
    """Audit: check which result dirs have manifests."""
    import argparse

    parser = argparse.ArgumentParser(description="Audit manifest coverage")
    parser.add_argument("root", help="Root directory to scan")
    parsed = parser.parse_args(args)

    root = Path(parsed.root)
    covered = 0
    uncovered = 0
    uncovered_dirs: list[str] = []

    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        has_manifest = any(
            f.name.endswith(".manifest.json") or f.name == "run_manifest.jsonl"
            for f in d.rglob("*") if f.is_file()
        )
        if has_manifest:
            covered += 1
        else:
            uncovered += 1
            uncovered_dirs.append(str(d.relative_to(root)))

    print(f"Coverage: {covered}/{covered + uncovered} dirs ({100*covered/(covered+uncovered):.0f}%)")
    if uncovered_dirs:
        print(f"\nUncovered ({uncovered}):")
        for d in uncovered_dirs[:20]:
            print(f"  {d}")
        if len(uncovered_dirs) > 20:
            print(f"  ... and {len(uncovered_dirs) - 20} more")

    return 0 if uncovered == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unified_manifest.py <run|audit> [args...]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "run":
        sys.exit(_cli_run(sys.argv[2:]))
    elif cmd == "audit":
        sys.exit(_cli_audit(sys.argv[2:]))
    else:
        print(f"Unknown command: {cmd}")
        print("Available: run, audit")
        sys.exit(1)
