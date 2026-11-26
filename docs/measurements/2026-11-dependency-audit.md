# Dependency and runtime audit — 2026-11-26

Companion to the [reliability recheck](2026-11-reliability-recheck.md). This audit covers
the package's direct and resolved transitive dependencies, build backend, development
tools, Docker runtime, offline runtime installer and repository-controlled CI pins.
It does not certify every dependency version allowed by a range, installed owner runtimes,
the external Ollama host, or GitHub's hosted runner software.

## Decisions and versions

The checkout had no repository dependency lockfile. It uses pip/PEP 517 and generates
hash-pinned requirements for each offline runtime bundle. Those mechanisms are preserved;
no second package manager or speculative transitive override is introduced. The fresh
baseline installation had already resolved current runtime packages. Raising minimums
therefore excludes older affected installations; it is not a claim that all local packages
changed version during this audit.

| Component | Previous declaration or observed version | New declaration / verified resolution |
| --- | --- | --- |
| Python in CI / Docker | Floating `3.12` / `python:3.12-slim` | `3.12.16`; explicit Debian trixie image and multi-platform digest |
| Pydantic | `>=2.9` | `>=2.14.2,<3`; current 2.14.4 |
| pydantic-settings | `>=2.6` | `>=2.16.3,<3`; current 2.16.3 |
| OpenAI SDK | `>=1.0` | `>=3.59.0,<4`; current 3.60.0 |
| HTTPX2 | Transitive through OpenAI, resolved 2.16.2 | Explicit `>=2.16.2,<3` for public transport-error handling; resolved version unchanged |
| MCP SDK | `>=1.0,<2.0` | `>=1.30.0,<2.0`; current 1.30.0 |
| Hatchling build backend | Unbounded | `>=1.33.0,<2`; isolated builds use 1.33.0 |
| Ruff | `>=0.16,<0.17`; baseline 0.16.10; hook 0.16.5 | `>=0.17.6,<0.18`; hook and environment 0.17.6 |
| mypy | `>=1.13,<2`; baseline 1.20.2 | `>=2.4.1,<2.5`; 2.4.1, strict check passes |
| Bandit | Separate unbounded installation | Dev/CI `>=1.9.4,<2`; 1.9.4 |
| pip in CI, Docker and new offline bundles | Image/ensurepip supplied 25.0.1 | `>=26.3,<27`; 26.3, hashed in each new bundle |
| pre-commit-hooks / yamllint / hadolint | 4.5.0 / 1.33.0 / 2.12.0 | 6.0.0 / 1.38.0 / 2.15.1, immutable revisions |
| actions/checkout / setup-python | Floating v4 / v5 | 7.0.1 / 7.1.0, immutable commits |
| Docker setup-buildx / build-push actions | Floating v3 / v5 | 4.4.1 / 7.4.0, immutable commits |
| Buildx CLI selected by CI | Runner-installed version or latest download | Explicit `v0.39.1`; official downloads verified and local build/export passed |
| Gitleaks | v2 JavaScript action wrapper | Standalone 8.30.1 with verified archive SHA-256 |

HTTPX 0.29.0, structlog 26.1.0 and sqlite-vec 0.1.9 are current compatible resolutions;
their existing ranges are retained. Existing dev ranges resolve pytest 9.2.0,
pytest-asyncio 1.5.0 and Hypothesis 6.170.17. There is no evidence requiring an application
migration for these packages. The hadolint hook now pins its actual container digest as
well as its Git revision. The old `check-yaml --safe` argument was invalid (exit 2);
safe parsing is already the hook's default, so the argument was removed.

HTTPX2 was already present in both checked OpenAI graphs. It is now a direct dependency
because malformed HTTP error-body reads can escape the SDK as public HTTPX2 transport
exceptions. Catching those explicitly avoids a broad exception handler or reliance on SDK
internals. The selected [HTTPX2 2.16.2 release](https://github.com/pydantic/httpx2/blob/main/src/httpx2/CHANGELOG.md)
restores runtime type aliases and includes 2.16.1's invalid-redirect-header correction;
this does not migrate Morgan's separate HTTPX-based embedding/MCP adapters.

Pinning the setup-buildx action alone did not pin its downloaded CLI; the explicit version
closes that gap. All four repository workflows explicitly limit `GITHUB_TOKEN` to
`contents: read`; none of their jobs requires repository write permissions. Buildx 0.39.1
was exercised from temporary plugin/config/state directories
against the existing engine, without replacing the owner's plugin. The Dockerfile frontend
retains Docker's [recommended stable `docker/dockerfile:1` channel](https://docs.docker.com/build/buildkit/frontend/)
for bug/security updates. That frontend and the action's default BuildKit daemon image
remain rolling upstream inputs; the CLI pin does not make the whole builder immutable.

## Security basis and applicability

- **MCP 1.30.0** includes fixes for
  [HTTP session exhaustion](https://github.com/modelcontextprotocol/python-sdk/security/advisories/GHSA-84m7-p3x7-pcfv)
  and [unbounded HTTP request bodies](https://github.com/modelcontextprotocol/python-sdk/security/advisories/GHSA-fmmv-w9g8-j3gc).
  The server paths matter to Morgan's HTTP surface. It also excludes versions affected by
  [OAuth credential handling](https://github.com/advisories/GHSA-qx49-fqc8-xw99) and
  [cross-origin redirect forwarding](https://github.com/advisories/GHSA-5h93-6whr-6q8j);
  Morgan does not enable that SDK OAuth/client flow. Morgan's separate chat redirect bug
  was reproduced and fixed in application code, as recorded in the reliability report.
- **Pydantic 2.14.2** closes
  [extra-key JSON validation bypass](https://github.com/pydantic/pydantic/security/advisories/GHSA-qpmf-fgh3-ff25)
  and [aliased TypedDict/dataclass validation bypass](https://github.com/pydantic/pydantic/security/advisories/GHSA-3r8g-cqwr-jr29).
  These are dependency protections, not demonstrated Morgan exploits. The
  [maintainer policy](https://github.com/pydantic/pydantic/security) supports the latest V2
  minor for security fixes.
- **Settings 2.16.3** fixes
  [secret-file symlink escape](https://github.com/pydantic/pydantic-settings/releases/tag/v2.16.3).
  Morgan does not configure a secrets directory. Its required python-dotenv floor also
  includes the security update described in
  [Settings 2.16.0](https://github.com/pydantic/pydantic-settings/releases/tag/v2.16.0).
  The old Settings 2.6 floor additionally predated the `NoDecode` API Morgan imports.
- **OpenAI 3.59.0** includes credential-header redaction in errors and request handling
  corrections documented in its
  [release notes](https://github.com/openai/openai-python/releases/tag/v3.59.0).
  Current 3.60.0 and the declared minimum are checked with real-socket adapter tests.
  This does not imply every earlier 1.x version has a published CVE.
- **Python 3.12.16** is a supported security release, including compromised CA distrust,
  archive handling and Expat fixes. Python 3.12 receives security fixes until October 2028.
  See the [official release](https://www.python.org/downloads/release/python-31216/).
  The macOS test interpreter remains 3.12.13: the installed uv catalogue had no 3.12.16
  macOS arm64 download. Linux checks use the patched container, without altering the host.
- **pip 26.3** removes six findings present in the image's 25.0.1: CVE-2025-8869,
  CVE-2026-13346, CVE-2026-3219, CVE-2026-6357, CVE-2026-8643 and CVE-2026-1703.
  They concern archive, wheel and index handling; for example the
  [malicious-index advisory](https://github.com/advisories/GHSA-qwm4-qh6w-59xr)
  is directly relevant to a wheel downloader's trust boundary.
  New offline bundles include pip in the same verified wheel manifest and install it
  without network resolution before installing Morgan. The builder also refuses an older,
  missing, prerelease or unsupported-major host pip before creating the bundle or resolving
  network dependencies; it does not upgrade the host automatically. Existing bundles keep their exact
  original artifacts and must be rebuilt to acquire the fix; no installed runtime is upgraded.

## Verification and outstanding findings

The unchanged baseline was **2112 passed, 4 skipped**. The reliability changes passed
**2181 tests, 4 skipped** on macOS before the final dependency/runtime edits; subsequent
focused checks and the final supported-runtime suite are recorded below rather than
silently treating that earlier run as the final tree.

| Check | Observed result |
| --- | --- |
| Ruff 0.17.6 lint / format | Pass; 335 Python files formatted |
| mypy 2.4.1, strict | Pass; 105 source files |
| Bandit 1.9.4 | Zero findings; existing nosec-comment warnings remain |
| `python -m pip check` | No broken requirements |
| `python -m build --outdir /tmp/morgan-reviewed-final-build` | sdist and wheel built with isolated Hatchling 1.33.0 |
| Exact committed runtime bundle, offline install and isolated check | Revision `d43888ddbb64573ca661ad0f5e3a2ab95922b456`; 106 package files verified, non-editable, pip 26.3, `pip check` passed |
| `pip-audit --cache-dir /tmp/morgan-recheck-pip-audit-cache --format json` | 81 installed Python distributions audited, zero known vulnerabilities; Morgan itself skipped because it is not published on PyPI |
| Build-tool graph: Hatchling 1.33.0, build 1.6.1, pip 26.3 | Nine resolved distributions audited, zero known vulnerabilities |
| Exact offline runtime graph | 38 distributions audited, zero known vulnerabilities; one unpublished Morgan skip |
| Declared security-minimum graph | 57 distributions audited, zero known vulnerabilities; one unpublished Morgan skip |
| Minimum-version provider/config/CLI/MCP/owner tests | 397 passed, one 15-second subprocess timeout in 1556.38s; unchanged failing case passed separately on minimum and current dependencies |
| Final streaming-error correction, current OpenAI 3.60.0 / minimum 3.59.0 | 89 provider tests passed on each graph; six real-socket SSE/read regressions failed before the fix |
| PR review: malformed error-body and client lifecycle corrections | 451 provider/app/strict-surface cases passed on current dependencies; 120 selected cases passed with minimum OpenAI 3.59.0 |
| PR review: overlapping canary revocation | 51 focused and 652 broader memory/import/migration/restore checks passed |
| Rebuilt final Docker image, installed package | 178 provider/lifecycle/cache cases passed in 8.67s, offline; all 105 installed source files match; CLI/MCP scope, persistence, conflict, snapshot and restore smoke passed |
| First Linux 3.12.16 full suite, four workers over host bind mount | 2147 passed, 40 failed, four skipped in 1422.87s; deadline/contention failures retained for serial investigation |
| Native Linux serial rerun of failed nodes, retry-budget probe and builder preflight | 43 passed, three failed in 324.37s; no selected node absent |
| Same native environment, unchanged baseline's three persistent nodes | Two passed; fact-writer contention reproduced in 40.97s |
| Final native Linux serial checks after the test-contract correction | 16 passed in 40.06s: both model-cycle cases, both writer races and all 12 database units |
| New CI dependency-job installation, isolated host environment | 63 distributions audited, zero known vulnerabilities; one unpublished Morgan skip |
| Actionlint / YAML / pre-commit config / Hadolint | Pass; 34 final scanner range/merge/failure/metadata fixtures pass (nine earlier baseline fixtures also passed) |

The pip-audit environment includes runtime, development and audit-tool dependencies.
Its clean result applies to the resolved versions and advisory database at this date;
it does not cover operating-system packages or JavaScript embedded in Actions.
The requirements-mode build scan initially hit a macOS `ensurepip` subprocess SIGABRT.
Creating a separate environment with the already available uv, resolving the same exact
build tools, and scanning its installed `site-packages` succeeded. This is an audit-tool
environment failure, not a Morgan regression.
The final container test setup initially selected a macOS Hypothesis wheel from the host's
download cache. Supplying the matching Linux ARM64 wheel allowed the offline installed-package
test run to execute and pass; no application assertion ran in that failed setup attempt.

The final application revision was exported with `scripts/runtime_bundle.py build --repo .
--revision d43888ddbb64573ca661ad0f5e3a2ab95922b456 --output <new-bundle>`, then installed
into a new private temporary directory using the matching explicit revision. Running
`python -I -m morgan_brain.runtime_check --expected-revision
d43888ddbb64573ca661ad0f5e3a2ab95922b456` from outside the checkout verified all 106 package
files and the non-editable origin. The install used only the bundle's verified wheels,
including pip 26.3. No runtime was activated. Artifact checks refer to this exact revision;
subsequent CI-only and report-only commits do not change its packaged application code.

The minimum-version test environment pins Pydantic 2.14.2, Settings 2.16.3, OpenAI 3.59.0,
MCP 1.30.0 and HTTPX 0.29.0; remaining transitives resolve current versions. Its only
failure was `test_reviewed_task_over_capacity_does_not_poison_recovery`: a local hash-backed
`owner_input status` subprocess exceeded the test helper's 15-second timeout during
concurrent host load. The same case passed unchanged on the minimum graph in 221.19s
and the current graph in 217.68s. No timeout was relaxed. This is not reported as a fully
green broad run or an exhaustive dependency matrix.

The first Linux run used four pytest workers while other host checks were active and
imported source through an OrbStack host bind mount. Its 40 failures comprise CLI helper
timeouts, bounded model startup/guardian waits and lock-startup polling, plus two SQLite
writers exhausting the configured five-second busy timeout. No atomicity violation was
reported by those writers. The same container separately passed all 26 new chat-status
tests; a retry-budget timing probe also exceeded its 2.3-second assertion under load.
The full run is preserved as a failed run, not relabelled green. Moving to a native
container filesystem, an ordinary user and serial execution cleared the guided/owner
timeouts and retry-budget failure. Two model-cycle readiness waits still exceeded their
test deadlines, then passed on the unchanged baseline; these are retained as separate
observations, not evidence of a model protocol change.

The fact-writer failure reproduced on the unchanged baseline in the same native environment.
The successful writer's 1,000 independently committed facts spanned about 30 seconds;
the other writer exhausted the default five-second busy wait. SQLite integrity checking
passed and no duplicate current fact keys existed. SQLite's busy handler does not promise
fair lock handoff between tight transaction loops. These atomicity tests already gave
each writer 120 seconds overall but implicitly required it to acquire every lock within
five. Both fact/vector race fixtures now explicitly use that existing 120-second test
budget; data and atomicity assertions and the outer deadlines remain intact. Production
`MORGAN_DB_BUSY_TIMEOUT_MS=5000` is unchanged. A separate real-connection test verifies
configured short-budget `SQLITE_BUSY`, no partial write and successful recovery after
the competing lock is released. This is a test-contract correction, not a claim that
arbitrary production contention can never exhaust the configured timeout.
The final 16-case native run passes without increasing any model/CLI deadline. This
resolves the observed affected checks; hosted CI supplies the subsequent whole-tree gate.

### Container findings

Trivy 0.76.2 used its database updated at `2026-11-26T19:04:08Z`, without ignoring
unfixed vulnerabilities. The initial patched-Python image still had pip 25.0.1 and
225 Debian package/advisory matches (101 distinct advisory IDs). Updating pip and the three available
Debian packages removes all six pip findings and the libzstd1 / passwd / login.defs
findings (CVE-2026-124851 and CVE-2026-135901).

The final local image is
`sha256:74c29bff0bfc677dc539d79371059534a4d8478025caff1298af329d644db4af`.
It still has **222 Debian matches, 99 distinct advisory IDs: 53 HIGH, 79 MEDIUM, 77 LOW,
13 UNKNOWN and zero CRITICAL**. No match has an available fixed trixie version in this
scan. Those 99 IDs comprise 88 CVEs and 11 Debian `TEMP-*` tracking identifiers. The
53 HIGH matches represent 13 distinct CVEs, repeated across binary packages:

| Source package | HIGH matches | Primary Debian tracking |
| --- | --- | --- |
| util-linux | 36 | [76642](https://security-tracker.debian.org/tracker/CVE-2026-76642), [78408](https://security-tracker.debian.org/tracker/CVE-2026-78408), [78409](https://security-tracker.debian.org/tracker/CVE-2026-78409), [78410](https://security-tracker.debian.org/tracker/CVE-2026-78410) |
| openssl | 6 | [107494](https://security-tracker.debian.org/tracker/CVE-2026-107494), [124709](https://security-tracker.debian.org/tracker/CVE-2026-124709) |
| ncurses | 4 | [2025-69720](https://security-tracker.debian.org/tracker/CVE-2025-69720) |
| sqlite3 | 3 | [119373](https://security-tracker.debian.org/tracker/CVE-2026-119373), [119374](https://security-tracker.debian.org/tracker/CVE-2026-119374), [119379](https://security-tracker.debian.org/tracker/CVE-2026-119379) |
| systemd | 2 | [16742](https://security-tracker.debian.org/tracker/CVE-2026-16742) |
| acl / perl | 1 each | [54369](https://security-tracker.debian.org/tracker/CVE-2026-54369), [9538](https://security-tracker.debian.org/tracker/CVE-2026-9538) |

Some trackers explicitly defer a stable-distribution update or mark the issue `no-dsa`.
This is not a blanket not-affected assessment. For example, the SQLite FTS5 advisory
requires a particular table configuration and negative rowids, and the OpenSSL advisory
above concerns QUIC. Morgan exploitability was not demonstrated; replacing SQLite,
OpenSSL or the distribution manually would require separate compatibility validation.

Trivy additionally reports three setuptools 70.3.0 matches from pip's vendor SBOM.
These are **not applicable to the inspected image**: no setuptools distribution or
vendored setuptools directory is installed, and the affected `package_index.py`,
`egg_info.py` and `setuptools/wheel.py` modules are absent. Only `pkg_resources` is retained,
consistent with [pip's vendoring rules](https://raw.githubusercontent.com/pypa/pip/26.3/pyproject.toml).
The relevant setuptools advisories are
[GHSA-5rjg-fvgr-3xxf](https://github.com/pypa/setuptools/security/advisories/GHSA-5rjg-fvgr-3xxf),
[GHSA-h35f-9h28-mq5c](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c)
and [GHSA-xmj3-9gf7-h52w](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w).
Raw matches remain in the [sanitized findings inventory](2026-11-dependency-findings.json).

### CI tooling findings

Latest upstream action releases are pinned to immutable commits. Their production npm/Yarn
lockfiles still have advisory matches: checkout has one vulnerable package (HIGH),
setup-python five (one HIGH, three MODERATE, one LOW); build-push has 33 advisory rows
(two CRITICAL), and setup-buildx 38 (two CRITICAL). These units are not interchangeable.
The Docker action findings include
[Handlebars template/AST concerns](https://github.com/advisories/GHSA-8r5x-fm3f-whwj)
and [prototype/template controls](https://github.com/advisories/GHSA-p8wg-vrv2-v86f).
Lockfiles are evidence of upstream dependency exposure, not proof that a vulnerable
function is present and reachable in the action's compiled JavaScript with this workflow.
The image-build workflow remains manual and does not push images. No custom action fork
or unsupported dependency override was introduced. All advisory IDs are in the inventory.

The latest Gitleaks action wrapper also reports 22 affected npm packages/dependents,
including one CRITICAL. Calling the checksum-verified official scanner directly removes
that JavaScript tree while retaining failure-on-findings and redaction. PR/push checks
scan changed history; manual/scheduled runs scan full history. Merge patches are included
without restricting traversal to first parents: a synthetic credential introduced only
by merge resolution escaped the old range and full-history modes, then failed both corrected
modes. Merged side-branch findings remain covered.

A zero-before push skips scanning only for an explicitly empty branch-creation event whose
head is proven already reachable from a different fetched default-branch ref. Other initial
or uncertain pushes scan all ancestors of the validated head, never unrelated refs. The
payload's commit list is not treated as complete because GitHub can truncate it. This is a
conservative fallback, not a claim to reconstruct every initial push's exact new range.

Scoped PR/push findings print only JSON-escaped commit/path/line/rule metadata, with no
matched text or commit message. Full-history and initial-head scans keep historical locations
private and print a local reproduction command. Temporary reports/logs are private and
removed on exit; no raw report is uploaded. Findings, incomplete scans, invalid refs and
invalid reports still fail. Wrapper PR comments and annotations are no longer produced.
A new pip-audit job checks runtime dependencies in the existing security workflow.

The scanner binary itself is **not vulnerability-free**. Official Gitleaks 8.30.1 was
built with Go 1.24.11. govulncheck 1.10.0 against `vuln.go.dev` reports 73 unique module
advisories, 72 also matched to binary symbols. Examples include
[Go archive allocation](https://pkg.go.dev/vuln/GO-2026-4869),
[RAR allocation](https://pkg.go.dev/vuln/GO-2025-4020) and
[compression parsing](https://pkg.go.dev/vuln/GO-2026-6865).
Binary symbol matches do not prove invocation during the configured local Git scan.
An upstream scanner rebuild/update remains required; building a private fork was deferred.

Current-source Gitleaks scanning is clean after two narrowly annotated synthetic test-key
constants. The original full-history run reported 15 redacted rows (14 unique scanner
fingerprints) across its baseline history. On the later refs, the same old options still
report 15 rows; merge-aware scanning reports 18 rows (17 fingerprints), adding only three
merge-commit findings. Exact historical-context review identifies those additions as two
repeated synthetic fixtures and one repeated false positive. Final triage is therefore
12 synthetic rows, three non-credential false positives and **three rows representing the
same two unresolved historical tokens:** one provider key and a duplicated browser-embedded
third-party token whose
authorization scope is unknown. Exact locations and rationale are retained in the private
audit artifacts; no payload or credential locator is republished here. No authentication
request was made to test either token. The owner must verify revocation or rotate any
still-valid credential before treating the history as cleared, consistent with
[GitHub's guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
No history was rewritten and no blanket allowlist was added. Manual/scheduled scans
continue failing on the documented historical findings; no compromise is asserted.

## Deferred changes

- **Host Docker toolchain:** installed Engine/CLI 29.8.2, Buildx 0.37.2 and Compose 5.5.0
  trail official 29.10.1 / 0.39.1 / 5.8.0. Engine 29.8.2 is affected by
  [CVE-2026-135124](https://github.com/moby/moby/security/advisories/GHSA-vfmq-r373-x89f),
  fixed in 29.10.1: a `docker cp` race into a running container with a read-only bind mount.
  This audit did not use `docker cp`. Updating the owner's OrbStack-managed host engine is
  outside the repository-only change scope. See the
  [Engine release notes](https://docs.docker.com/engine/release-notes/29/),
  [Buildx release](https://github.com/docker/buildx/releases/tag/v0.39.1) and
  [Compose release](https://github.com/docker/compose/releases/tag/v5.8.0).
- **MCP 2.3.0:** a separate migration from HTTPX to HTTPX2 transport objects and changed
  SDK APIs. OpenAI's use of HTTPX2 does not migrate the MCP adapters. The maintained 1.x
  security line supplies the relevant fixes;
  see the [SDK roadmap](https://github.com/modelcontextprotocol/python-sdk/blob/main/ROADMAP.md).
- **Python feature-version migration:** 3.12.16 retains the tested ABI and supported
  security line. Python 3.13–3.15, Windows and alternate sqlite-vec wheel combinations
  have not been accepted by this audit.
- **Whole-range guarantees:** minimum direct security versions and the current resolved
  graph are checked; no exhaustive matrix of all allowed transitive versions is claimed.
  Build/development installs continue resolving ranges. Production offline bundles retain
  exact wheel hashes; a reviewed base image plus `apt-get upgrade` is not a reproducible
  final container until the resulting image digest is recorded.
- **FastMCP cwd dotenv:** the existing invalid-UTF-8 `.env` startup failure remains in
  SDK 1.30.0. No supported constructor opt-out exists in that SDK. A global monkeypatch or
  a transport-major migration would exceed the compatible fix and is not hidden here.

No container was pushed or activated, no owner runtime was changed, and no service was
installed. Publishing source changes does not deploy the runtime.
