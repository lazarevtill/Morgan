"""Skills module — implements ``interfaces.SkillEngine``.

Responsibility: discover/version/select markdown skills and inject the active best_skill.md
into reasoning context. Training is offline (learning-worker); request path only reads.
Service: brain-api (select/get) + learning-worker (deploy after validation). Phase: 3.

Planned files: registry.py, executor.py, bundled/*.md (conversation, empathy, research,
coding, planning, calendar).
"""
