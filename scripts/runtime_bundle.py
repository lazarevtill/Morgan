"""Build a trusted Git revision into an offline wheel bundle; install into a fresh runtime."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def run(args: list[str], *, cwd: Path | None = None) -> bytes:
    """Run a fixed argv with finite timeout; no shell, editable install or global mutation."""
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, timeout=600).stdout  # noqa: S603


def sha(path: Path) -> str:
    """Hash exact archive bytes without relying on filenames or package versions."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def commit(value: str) -> str:
    """Require an explicit immutable full revision, not a mutable branch name."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("revision requires a full lowercase commit SHA")
    return value


def build(repo: Path, revision: str, output: Path) -> dict[str, object]:
    """Export only committed sources, stamp them, then freeze resolved binary dependencies."""
    commit(revision)
    actual = run(["git", "rev-parse", f"{revision}^{{commit}}"], cwd=repo).decode().strip()
    if actual != revision:
        raise ValueError("Git did not resolve the requested commit")
    output.mkdir(mode=0o700)  # Refuse reuse, including a partial prior build.
    wheels = output / "wheels"
    wheels.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix="morgan-build-") as temporary:
        source = Path(temporary) / "source"
        source.mkdir()
        archive = Path(temporary) / "source.tar"
        archive.write_bytes(run(["git", "archive", "--format=tar", revision], cwd=repo))
        with tarfile.open(archive) as exported:
            exported.extractall(source, filter="data")
        package = source / "morgan_brain"
        manifest = {
            "version": 1,
            "revision": revision,
            "files": {
                str(p.relative_to(package)): sha(p)
                for p in sorted(package.rglob("*"))
                if p.is_file()
            },
        }
        if (package / "_runtime_build.json").exists():
            raise ValueError("source must not contain a preexisting build stamp")
        (package / "_runtime_build.json").write_text(json.dumps(manifest, sort_keys=True))
        run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "wheel",
                "--no-deps",
                "--index-url",
                "https://pypi.org/simple",
                "--wheel-dir",
                str(wheels),
                str(source),
            ]
        )
    candidates = list(wheels.glob("morgan_brain-*.whl"))
    if len(candidates) != 1:
        raise ValueError("build must produce one Morgan wheel")
    run(
        [
            sys.executable,
            "-I",
            "-m",
            "pip",
            "--isolated",
            "download",
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            "--dest",
            str(wheels),
            str(candidates[0]),
        ]
    )
    requirements = []
    archives = {}
    names = set()
    for wheel in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(wheel) as zipped:
            metadata = [name for name in zipped.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("wheel requires one distribution metadata record")
            info = email.message_from_bytes(zipped.read(metadata[0]))
        name, version = info["Name"], info["Version"]
        if not name or not version or re.fullmatch(r"[A-Za-z0-9_.+-]+", name + version) is None:
            raise ValueError("invalid dependency metadata")
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in names:
            raise ValueError("bundle contains duplicate distribution versions")
        names.add(normalized)
        digest = sha(wheel)
        archives[wheel.name] = digest
        requirements.append(f"{name}=={version} --hash=sha256:{digest}")
    lock = output / "requirements.txt"
    lock.write_text("\n".join(requirements) + "\n")
    receipt = {
        "version": 1,
        "revision": revision,
        "wheels": archives,
        "requirements_sha256": sha(lock),
        "python": sys.version,
        "platform": sys.platform,
        "machine": platform.machine(),
        "python_tag": sys.implementation.cache_tag,
        "morgan_wheel": candidates[0].name,
    }
    (output / "bundle.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def install(bundle: Path, revision: str, destination: Path) -> dict[str, object]:
    """Verify bundle hashes before creating a fresh environment; never modify active runtimes."""
    commit(revision)
    receipt = json.loads((bundle / "bundle.json").read_text())
    if receipt.get("version") != 1 or receipt.get("revision") != revision:
        raise ValueError("bundle revision differs from requested revision")
    if (receipt.get("platform"), receipt.get("machine"), receipt.get("python_tag")) != (
        sys.platform,
        platform.machine(),
        sys.implementation.cache_tag,
    ):
        raise ValueError("bundle requires its original OS, architecture and Python minor version")
    lock = bundle / "requirements.txt"
    if sha(lock) != receipt["requirements_sha256"]:
        raise ValueError("requirements lock changed")
    for name, digest in receipt["wheels"].items():
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError("invalid wheel filename")
        if sha(bundle / "wheels" / name) != digest:
            raise ValueError("wheel bytes changed")
    destination.mkdir(mode=0o700)  # Leave partial installs for inspection; restore into a new path.
    run([sys.executable, "-I", "-m", "venv", str(destination)])
    python = destination / "bin/python"
    run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-index",
            "--only-binary=:all:",
            "--require-hashes",
            "--find-links",
            str(bundle / "wheels"),
            "-r",
            str(lock),
        ],
        cwd=destination,
    )
    run([str(python), "-I", "-m", "pip", "check"], cwd=destination)
    check = run(
        [str(python), "-I", "-m", "morgan_brain.runtime_check", "--expected-revision", revision],
        cwd=destination,
    )
    for module in ("owner_input", "model_cycle", "model_guardian"):
        # Import only: model_guardian has no help CLI. No memory or inference starts.
        run([str(python), "-I", "-c", f"import morgan_brain.{module}"], cwd=destination)
    verified = json.loads(check)
    (destination / "morgan-runtime.json").write_text(json.dumps(verified, indent=2))
    return verified


def main() -> None:
    """Select build or offline install; a failed attempt never activates a runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--repo", type=Path, required=True)
    builder.add_argument("--revision", required=True)
    builder.add_argument("--output", type=Path, required=True)
    installer = commands.add_parser("install")
    installer.add_argument("--bundle", type=Path, required=True)
    installer.add_argument("--revision", required=True)
    installer.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(args.repo.resolve(), args.revision, args.output.absolute())
        else:
            result = install(args.bundle.resolve(), args.revision, args.destination.absolute())
        print(json.dumps(result))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"Runtime operation failed: {error}\n")


if __name__ == "__main__":
    main()
