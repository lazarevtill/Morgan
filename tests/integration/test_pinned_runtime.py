"""Actual wheels/venvs with a tiny fixture package; no model, service or real owner data."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def command(*args, cwd=None, env=None):
    return subprocess.run(
        list(args), cwd=cwd, env=env, capture_output=True, text=True, timeout=120, check=True
    )


def helper():
    spec = importlib.util.spec_from_file_location(
        "runtime_bundle", ROOT / "scripts/runtime_bundle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("runtime-artifacts")
    repo = root / "repo"
    repo.mkdir()
    package = repo / "morgan_brain"
    package.mkdir()
    (package / "__init__.py").write_text('"""Fixture only."""\n')
    (package / "runtime_check.py").write_bytes(
        (ROOT / "morgan_brain/runtime_check.py").read_bytes()
    )
    for name in ("owner_input.py", "model_cycle.py", "model_guardian.py"):
        (package / name).write_text('"""Fixture entrypoint; no real inference or tasks."""\n')
    (repo / "pyproject.toml").write_text("""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[project]
name = "morgan-brain"
version = "0.0.0"
requires-python = ">=3.12"
[tool.hatch.build.targets.wheel]
packages = ["morgan_brain"]
""")
    command("git", "init", "-q", str(repo))
    command("git", "add", ".", cwd=repo)
    command(
        "git",
        "-c",
        "user.name=Runtime Test",
        "-c",
        "user.email=runtime@example.invalid",
        "commit",
        "-qm",
        "fixture",
        cwd=repo,
    )
    revision = command("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    # A dirty checkout must not leak into an allegedly pinned build.
    (package / "owner_input.py").write_text('raise RuntimeError("dirty source imported")\n')
    artifact = root / "bundle"
    helper().build(repo, revision, artifact)
    return artifact, revision


def verify(runtime, revision, *, cwd=None, env=None):
    return command(
        str(runtime / "bin/python"),
        "-I",
        "-m",
        "morgan_brain.runtime_check",
        "--expected-revision",
        revision,
        cwd=cwd,
        env=env,
    )


def test_clean_install_ignores_cwd_pythonpath_and_dirty_checkout(bundle, tmp_path):
    artifact, revision = bundle
    target = tmp_path / "runtime"
    result = helper().install(artifact, revision, target)
    assert result["revision"] == revision and result["editable"] is False
    shadow = tmp_path / "shadow/morgan_brain"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text('raise RuntimeError("shadow import")\n')
    env = {**os.environ, "PYTHONPATH": str(shadow.parent)}
    verified = json.loads(verify(target, revision, cwd=shadow.parent, env=env).stdout)
    assert Path(verified["package_root"]).is_relative_to(target)
    command(
        str(target / "bin/python"),
        "-I",
        "-m",
        "morgan_brain.owner_input",
        cwd=shadow.parent,
        env=env,
    )
    assert (target / "morgan-runtime.json").exists()
    with pytest.raises(ValueError, match="revision"):
        helper().install(artifact, "0" * 40, tmp_path / "wrong")
    assert not (tmp_path / "wrong").exists()


def test_changed_package_fails_and_restore_uses_new_path(bundle, tmp_path):
    artifact, revision = bundle
    old = tmp_path / "old"
    info = helper().install(artifact, revision, old)
    source = Path(info["package_root"]) / "owner_input.py"
    source.write_text('raise RuntimeError("changed installed code")\n')
    with pytest.raises(subprocess.CalledProcessError):
        verify(old, revision)
    with pytest.raises(FileExistsError):
        helper().install(artifact, revision, old)
    restored = tmp_path / "restored"
    helper().install(artifact, revision, restored)
    assert json.loads(verify(restored, revision).stdout)["revision"] == revision
    assert "changed installed code" in source.read_text()


def test_partial_install_does_not_publish_ready_marker(bundle, tmp_path, monkeypatch):
    artifact, revision = bundle
    module = helper()
    original = module.run

    def interrupted(args, **kwargs):
        if "install" in args:
            raise subprocess.CalledProcessError(29, args)
        return original(args, **kwargs)

    monkeypatch.setattr(module, "run", interrupted)
    partial = tmp_path / "partial"
    with pytest.raises(subprocess.CalledProcessError):
        module.install(artifact, revision, partial)
    assert partial.exists() and not (partial / "morgan-runtime.json").exists()
    monkeypatch.setattr(module, "run", original)
    restored = tmp_path / "restored"
    module.install(artifact, revision, restored)
    assert (restored / "morgan-runtime.json").exists()
    assert not (partial / "morgan-runtime.json").exists()


@pytest.mark.parametrize("file", ["requirements.txt", "wheel"])
def test_changed_bundle_is_refused_before_creating_runtime(bundle, tmp_path, file):
    import shutil

    artifact, revision = bundle
    copied = tmp_path / "bundle"
    shutil.copytree(artifact, copied)
    changed = (
        copied / "requirements.txt"
        if file == "requirements.txt"
        else next((copied / "wheels").glob("*.whl"))
    )
    changed.write_bytes(changed.read_bytes() + b"tampered")
    destination = tmp_path / "runtime"
    with pytest.raises(ValueError, match="changed"):
        helper().install(copied, revision, destination)
    assert not destination.exists()
