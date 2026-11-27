# Development

Morgan requires Python 3.12+. CI uses Python 3.12.16. Work from a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The project uses pip and PEP 517. There is no repository dependency lockfile. Deployable
runtime bundles generate their own exact, hash-pinned requirements; see
[pinned runtimes](docs/decisions/pinned-runtime.md).

## Checks

```bash
python -m pytest -q
ruff check .
ruff format --check .
mypy morgan_brain
bandit -c pyproject.toml -r morgan_brain
```

The ordinary test suite uses synthetic data and skips live-model tests. Use a scratch data
directory and explicitly configured endpoints for `python -m pytest tests/memory_quality --live`.
Hash embeddings verify storage and wiring, not semantic retrieval. Test totals belong to a
specific revision; see [measurement records](docs/measurements/README.md).

For storage or lifecycle changes, run the relevant integration tests as well as unit tests.
Use [AGENTS.md](AGENTS.md) for the invariants that tests must preserve. Never run destructive
checks against the owner's database or activate an editable checkout as a deployed runtime.

## Documentation

- Put installation and operation instructions in the guides; link to them from README.
- Record behavioral decisions under `docs/decisions/`, with the implementation boundary,
  recovery behavior and relevant tests. Distinguish implemented behavior from proposals.
- Record measurements under `docs/measurements/`: revision, environment, command, dataset,
  result, failures and limits. Preserve raw outputs, frozen cases and rejected candidates.
- Label illustrative output as an example. Do not present it as a test receipt.
- Describe the concrete behavior first. Remove promotional claims, repeated caveats and
  implementation chronology that does not help the reader use or assess the result.
- Keep prose within 100 columns where practical. Tables and code may be wider. Follow
  `.markdownlint.yaml`; validate local links and command examples when editing them.
- Keep historical research and archived designs dated. Update navigation or add a correction
  note instead of silently replacing historical results with current numbers.

Documentation-only changes need link, formatting and relevant example checks. Run application
tests when a changed example or claim depends on behavior that has not already been verified.
