# morgan-brain

A self-hosted, privacy-first **personal assistant that knows and learns from you**.

Built on the MAPLE decomposition (Memory ≠ Learning ≠ Personalization) and SkillOpt
(skills as trainable markdown). See [`docs/superpowers/specs/2026-06-07-morgan-brain-design.md`](../docs/superpowers/specs/2026-06-07-morgan-brain-design.md)
for the full design, and [`docs/ARCHITECTURE_V2.md`](../docs/ARCHITECTURE_V2.md) for background.

> The previous monolithic implementation is archived in the git branch/tag
> `legacy/v0.0.3-monolith` and is the source for selectively ported code.

## Topology (3 services, one package)

| Service | Role | Status |
|---------|------|--------|
| `brain-api` | The request path — Perception → Personalization → Memory → Skills → Reasoning → Tools | active |
| `learning-worker` | Async: trait extraction, user-model, SkillOpt training, consolidation, pattern mining | active |
| `perception-gpu` | Voice/vision (Whisper, Wav2Vec2). Interface defined; **not yet built** | deferred |

All three run from the single `morgan_brain` package. Modules talk over typed Protocols and an
event bus whose in-process and Redis-Streams backends share one interface — so any module can be
promoted to its own service without code changes.

## Layout

```
morgan_brain/
  config.py          # one MORGAN_-prefixed settings source
  interfaces/        # Protocols — the contracts every module implements
  models/            # shared domain models (user_id-keyed)
  bus/               # in-proc + Redis Streams event bus (one interface)
  security/          # MemoryGate + unified PermissionMode/PermissionGate
  modules/           # perception, memory, learning, personalization,
                     # reasoning, skills, tools, mcp, proactivity
  core/              # thin cognitive-loop orchestrator
  apps/              # brain_api, learning_worker, perception_gpu entrypoints
clients/cli/         # thin terminal client
tests/               # unit, integration, memory_quality (LoCoMo/LongMemEval-style)
```

## Quick start (dev)

```bash
cd morgan-brain
cp .env.example .env
pip install -e ".[dev]"
docker compose up -d redis qdrant            # infra
python -m morgan_brain.apps.brain_api        # request-path service
python -m morgan_brain.apps.learning_worker  # async worker
```

## Status

Phase 0 (foundation skeleton). See the design spec §14 for the phase plan.
