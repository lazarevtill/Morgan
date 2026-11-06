# A pinned, non-editable local runtime

The test Python imported an older editable checkout under `python -I`; `owner_input` did not
exist there. Starting from the right cwd hid that defect, while the isolated model guardian
could import a different checkout. A deployment must not depend on cwd or a moving editable path.

`scripts/runtime_bundle.py` builds an explicitly trusted **full Git commit**, not the working
tree or a branch name. It exports that commit, records the revision and package-file hashes in
`_runtime_build.json`, builds an ordinary wheel, and materializes its resolved binary dependencies.
The bundle contains wheel bytes, pinned `requirements.txt` with SHA256 hashes, and `bundle.json`.
Dirty/untracked working-tree files are excluded. No installed environment is changed by build.

```sh
python scripts/runtime_bundle.py build --repo /absolute/Morgan \
  --revision FULL_COMMIT_SHA --output /absolute/new-bundle
python scripts/runtime_bundle.py install --bundle /absolute/new-bundle \
  --revision FULL_COMMIT_SHA --destination /absolute/new-runtime
/absolute/new-runtime/bin/python -I -m morgan_brain.runtime_check \
  --expected-revision FULL_COMMIT_SHA
/absolute/new-runtime/bin/python -I -m morgan_brain.owner_input --help
```

Use Python 3.12+ with pip to operate the builder. The commit must already exist in that local
Git repository and contain `runtime_check`; build never fetches refs or guesses a revision.
Build can download build tools and binary dependencies from public PyPI. It is an explicit code
build of a trusted repository, not a sandbox for arbitrary build backends. Wheel resolution
happens once: preserve and trust this resulting bundle. Subsequent installation is offline,
uses `--no-index --only-binary --require-hashes`, and cannot re-resolve newer package versions.
The initial build backend and dependency resolution are not a universal reproducible-build lock;
this promises repeatable installation from the same saved artifacts, not identical wheels from
arbitrary future PyPI resolutions. No signing/remote attestation is claimed by local checksums.

The installer refuses another OS, architecture or Python minor version. The original Python
interpreter/stdlib is still a host prerequisite; it is not copied into a self-contained OS image.
The venv's interpreter may point to the host Python, so removal/upgrading of that base requires
building/verifying a replacement runtime. Native dependency compatibility is scoped to the
materialized wheels, not every future macOS release.

## Verification and recovery

Installation requires a nonexistent destination. It creates an isolated venv without system
site packages, installs exact artifacts, runs `pip check`, verifies Morgan's installed revision,
origin and package-file hashes, then imports owner_input/model_cycle/model_guardian under `-I`
outside the repository. Only after success does it write `morgan-runtime.json`. Neither imports
nor verification call a model, open personal memory, install a service or send a notification.

`runtime_check --expected-revision` rejects a wrong revision, editable/outside-prefix origin,
altered/missing package files and unexpected Python modules. The build stamp and bundle receipt
must come from a trusted build; somebody able to rewrite both code and manifests under the same
OS account is outside this integrity check. It verifies Morgan's package, not arbitrary later
mutations to every dependency or the operating system. Dependency versions/hashes are enforced
at install; `pip check` checks compatibility, not ongoing whole-environment tamper resistance.

No `current` symlink or service is activated automatically. The operator explicitly chooses the
verified runtime's absolute Python path. Never rename a populated venv: generated console scripts
can contain its original absolute path. A failed install leaves a partial directory for inspection
and no ready marker. Recovery installs the same bundle into another **new** absolute directory,
verifies it, and changes only the caller's selected executable. The previous runtime and data
remain intact. The installer never deletes runtimes, modifies another project or migrates memory.
A ready marker is evidence of a completed check, not a substitute for re-running verification.

## Tests and continuous-operation boundary

`test_pinned_runtime.py` builds a small fixture wheel with the actual verifier and uses real
venvs/pip. It covers committed-vs-dirty source, cwd/PYTHONPATH isolation, wrong revision,
changed package/wheel/lock, partial install without a ready marker and recovery into a new path.
The fixture's owner/model entrypoints are inert, not an inference or full-agent measurement.
A separate real Morgan bundle/install/proposal-free owner-input pilot must be recorded for the
selected host before calling that runtime operational.

A pinned runtime fixes import consistency; it does not create a 24/7 agent. A concrete next
local channel is the existing macOS native-dialog adapter plus a private explicit-message inbox.
A driver must carry message IDs, render exact review fields, distinguish dismiss/timeout from
confirmation, and pass only the reviewed digest to owner_input. The existing alert acknowledgement
means interaction only; it must not be silently promoted to task approval. An OS-account session,
not a typed owner label, remains the local trust boundary.

For eventual after-login operation, use an explicitly installed user LaunchAgent invoking that
specific verified runtime and bounded driver, with separate persistent STOP, pilot expiry and
finite model cycles. The driver/review UI still needs implementation and an end-to-end GUI test;
no executable for that complete loop is claimed here. A Mac asleep/offline/at the FileVault login
screen is not continuously available. Phone delivery requires an explicitly configured recipient
and channel (e.g. the existing reviewed ntfy adapter); app presence alone is not configuration.
No service, channel grant, credentials or paid process is installed by these build tools.

The [Mac measurement](../measurements/2026-11-pinned-runtime.json) built the real Morgan wheel
at `a27c6ac9403348deb0fec9cdd5ea8ef4373e9530`, installed its dependencies, and verified 98 package
files from site-packages. Fifteen commands outside the repository, with an old editable checkout
in PYTHONPATH, exercised synthetic task creation, renewal and STOP, then restored a second runtime
from the same offline bundle and read the same plan/STOP state. Total scenario including recovery
was 8.380 seconds. This proves the former editable-origin blocker is removed for that artifact;
it does not measure inference, an actual human acknowledgement or continuous operation.
