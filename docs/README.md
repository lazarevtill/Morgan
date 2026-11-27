# Documentation

Start with the guide for the task you are doing. Current guides describe the checked-in code;
measurements and research describe the revision and date recorded in each report.

## Use and operate Morgan

| Task | Document |
| --- | --- |
| Install, configure models and use CLI/MCP | [Wiring and running](WIRING.md) |
| Diagnose configuration, providers and stored vectors | [Diagnostics](DIAGNOSTICS.md) |
| Protect data, back up, restore and upgrade | [Operations](OPERATIONS.md) |
| Understand current limits and development priorities | [Roadmap](ROADMAP.md) |
| Develop and verify a change | [Contributing](../CONTRIBUTING.md) |

## Integrate a caller

| Contract | Document |
| --- | --- |
| Storage guarantees and caller responsibilities | [Memory guarantees](MEMORY_GUARANTEES.md) |
| Immutable source corrections and effective time | [Revision client](REVISION_CLIENT.md) |
| Exact, bounded context inspection | [Continuation context](CONTINUATION_CONTEXT.md) |
| Offline inspection example | [Zero-model continuation](ALWAYS_ON_MEMORY_EXAMPLE.md) |
| Requirements for model-generated answers | [Reader capability](READER_CAPABILITY.md) |
| Components and data flow | [Architecture](ARCHITECTURE.md) |

## Decisions and evidence

- [Decision index](decisions/README.md): contracts for memory, models, tasks, planning and owner
  input.
- [Measurement index](measurements/README.md): tests, experiments, raw evidence and known limits.
- [Research index](research/README.md): dated comparisons and proposals, separate from implemented
  APIs.
- [Archive](archive/README.md): designs for retired versions; not current operating instructions.

Repository agents use [AGENTS.md](../AGENTS.md). The packaged
[Morgan usage skill](../morgan_brain/surfaces/cli/skill/SKILL.md) guides external coding agents.
