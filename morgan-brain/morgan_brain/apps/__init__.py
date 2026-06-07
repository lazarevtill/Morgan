"""Service entrypoints. One package, three processes:

* brain_api        — the request path (FastAPI)
* learning_worker  — async intelligence extraction + SkillOpt
* perception_gpu   — voice/vision (deferred; interface only)
"""
