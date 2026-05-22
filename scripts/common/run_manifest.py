from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def snapshot_files(root: Path, patterns: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                out[str(path)] = path.stat().st_mtime
    return out


def changed_files(before: dict[str, float], after: dict[str, float]) -> list[str]:
    changed: list[str] = []
    for path, mtime in after.items():
        if path not in before or before[path] != mtime:
            changed.append(path)
    return sorted(changed)


def write_manifest(
    path: Path,
    run_type: str,
    changed: list[str],
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_type": run_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "changed_files": changed,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
