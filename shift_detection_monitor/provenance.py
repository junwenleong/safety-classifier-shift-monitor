"""Provenance manifest generation — compatibility shim.

This module previously contained the full manifest-building logic. That has been
consolidated into unified_manifest.py. This file re-exports the public API so
existing imports (``from shift_detection_monitor.provenance import write_manifest``)
continue to work without modification.
"""

from .unified_manifest import build_manifest, write_manifest

__all__ = ["build_manifest", "write_manifest"]
