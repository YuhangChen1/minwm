from __future__ import annotations

import os
from pathlib import Path


def replace_with_link(source: str | Path, destination: str | Path) -> Path:
    """Atomically point destination at source without duplicating checkpoint bytes."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source.parent != destination.parent:
        raise ValueError("Checkpoint aliases must live beside their source")
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        os.symlink(source.name, temporary)
    os.replace(temporary, destination)
    return destination
