"""Verify an installed, revision-stamped wheel independently of cwd and PYTHONPATH."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path


def verify(revision: str) -> dict[str, object]:
    """Fail closed on editable origins, wrong revisions, missing or altered package files."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("expected revision must be a full lowercase Git commit SHA")
    root = Path(__file__).resolve().parent
    dist = importlib.metadata.distribution("morgan-brain")
    if root != Path(str(dist.locate_file("morgan_brain"))).resolve() or not root.is_relative_to(
        Path(sys.prefix).resolve()
    ):
        raise ValueError("Morgan must be installed inside this runtime, not an editable checkout")
    direct = dist.read_text("direct_url.json")
    if direct and json.loads(direct).get("dir_info", {}).get("editable"):
        raise ValueError("editable installation is not a pinned runtime")
    manifest = json.loads((root / "_runtime_build.json").read_text())
    if manifest.get("revision") != revision or manifest.get("version") != 1:
        raise ValueError("installed revision does not match the expected build")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("build has no package file manifest")
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid package manifest path")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("package file escaped its installation")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"package file differs from build: {name}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*.py")}
    if actual != {name for name in files if name.endswith(".py")}:
        raise ValueError("installed Python sources differ from the build manifest")
    for required in ("owner_input.py", "model_cycle.py", "model_guardian.py", "runtime_check.py"):
        if required not in files:
            raise ValueError(f"build is missing required entrypoint: {required}")
    return {
        "revision": revision,
        "package_root": str(root),
        "python": sys.executable,
        "files_verified": len(files),
        "editable": False,
    }


def main() -> None:
    """Read integrity evidence only; never open memory, a provider or a model process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.expected_revision)))
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(2, f"Runtime verification failed: {error}\n")


if __name__ == "__main__":
    main()
