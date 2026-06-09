"""Skills module — implements ``interfaces.SkillEngine``.

Responsibility: discover/version/select markdown+frontmatter skills (trigger-matched) and inject
the active skill body into reasoning context. Training is offline (learning-worker, champion-
versioned via the optimizer); the request path only reads.
Service: brain-api (select/get; exposed at GET/POST /api/skills) + learning-worker (deploy after
the eval gate). Built.

Files: registry.py (SkillRegistry), frontmatter.py, bundled/*.md (conversation, empathy, research,
coding, planning).
"""
