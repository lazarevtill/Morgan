"""Learning module — implements ``interfaces.Learner``.

Responsibility: asynchronous intelligence extraction. Extract facts/preferences/behaviors from
completed sessions, maintain the stable UserModel, run SkillOpt training behind a validation
gate, consolidate (dedup/decay/curate), mine behavioral patterns for proactivity.
Service: learning-worker (off the request path). Phase: 2 (extraction + UserModel) · 3 (SkillOpt).

Planned files: extractors/{trait,preference,pattern}.py, user_model/model.py,
consolidation/{consolidator,decay}.py, skillopt/{trainer,trajectory,evaluator,registry}.py.
"""
