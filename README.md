# Morgan

A self-hosted, privacy-first **personal assistant that knows and learns from you** — every
interaction makes it know you better, and that knowledge changes the next response.

> **Fresh start (2026-06-07).** This repo was reset to a greenfield design. The previous
> monolithic implementation is archived in the git branch/tag **`legacy/v0.0.3-monolith`**.

## Where things are

| Path | What |
|------|------|
| **`morgan-brain/`** | The implementation (one package, three services). Start at its [README](morgan-brain/README.md). |
| **`docs/superpowers/specs/2026-06-07-morgan-brain-design.md`** | The design authority. |
| **`docs/ARCHITECTURE_V2.md`** | Background and rationale. |
| **`CLAUDE.md`** | Guidance for working in the repo. |

## Idea in one paragraph

Built on **MAPLE** (Memory ≠ Learning ≠ Personalization — three mechanisms on three timescales)
and **SkillOpt** (skills as trainable markdown). A request-path **`brain-api`** perceives,
personalizes, recalls memory, applies skills, and reasons; an async **`learning-worker`** extracts
who you are into a stable user model and improves skills off the critical path; a deferred
**`perception-gpu`** adds voice/vision later behind the same interface. Memory is bi-temporal
(facts evolve, they don't overwrite), all of it gated and `user_id`-keyed for privacy and
multi-tenant readiness.

## Quick start

```bash
cd morgan-brain
cp .env.example .env
pip install -e ".[dev]"
docker compose up -d redis qdrant
python -m morgan_brain.apps.brain_api   # then GET http://localhost:8080/health
```

See [`morgan-brain/README.md`](morgan-brain/README.md) for the full layout and the design spec
for the phase plan.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE).
