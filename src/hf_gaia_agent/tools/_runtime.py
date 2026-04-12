"""Runtime workspace helpers for tools that need scratch directories."""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

_RUNTIME_ROOT_ENV = "HF_GAIA_RUNTIME_DIR"


def runtime_root() -> Path:
    """Return a writable directory for ephemeral tool artifacts."""
    configured = os.getenv(_RUNTIME_ROOT_ENV, "").strip()
    root = Path(configured).expanduser() if configured else Path.cwd() / ".runtime-artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


@contextmanager
def runtime_workspace(prefix: str) -> Iterator[Path]:
    """Create and clean up an isolated scratch directory under the runtime root."""
    workspace = runtime_root() / f"{prefix}{uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
