"""Run provenance capture.

Addresses the reproducibility failure mode where a result file cannot be traced
back to the code, package versions, model weights, or API model snapshot that
produced it. Emit a manifest alongside every result file so that a later re-run
producing different numbers can be diagnosed (code moved? transformers upgraded?
HF weights re-pulled? API model silently swapped?) instead of guessed at.

Usage:
    from shift_detection_monitor.provenance import write_manifest
    write_manifest("results/factorial_results.jsonl")
    # -> writes results/factorial_results.jsonl.manifest.json

Optionally attach extra provenance (HF revisions, API fingerprints):
    write_manifest(out, extra={"api_models": API_MODELS_SEEN})
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Packages whose version silently changes model/scoring behaviour.
_KEY_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "pydantic",
    "torch",
    "transformers",
    "tokenizers",
    "openai",
    "datasets",
)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for pkg in _KEY_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


def build_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a provenance manifest describing the current run environment."""
    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": bool(dirty) if dirty is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    if extra:
        manifest["extra"] = extra
    return manifest


def write_manifest(
    result_path: str | Path, extra: dict[str, Any] | None = None
) -> Path:
    """Write a manifest next to ``result_path`` and return the manifest path."""
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = result_path.with_name(result_path.name + ".manifest.json")
    manifest_path.write_text(json.dumps(build_manifest(extra), indent=2))
    return manifest_path
