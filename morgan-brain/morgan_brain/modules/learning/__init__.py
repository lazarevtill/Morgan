"""Learning module — implements ``interfaces.Learner``.

Responsibility: asynchronous intelligence — capture signals, maintain the UserModel, consolidate
episodics into bi-temporal facts, and run the eval-gated champion-preprompt optimizer.
Service: learning-worker (off the request path). Built.

Implementation lives in the sibling top-level package, not here: ``morgan_brain.learning``
(recorder.py + signals.py — signal capture; learner.py + profile.py — UserModel;
consolidation.py — facts; champion_trainer.py + optimizer.py — GEPA optimizer). The eval gate is
``morgan_brain.eval``; the prompt-registry/optimizer seam is ``morgan_brain.learning_lifecycle``.
"""
