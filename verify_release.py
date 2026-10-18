"""Verify the distributed source/input/data checksums with standard Python."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError("Invalid path outside release root")
        if not path.is_file():
            failures.append(f"Missing: {entry['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(f"Changed: {entry['path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(manifest['files'])} distributed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
