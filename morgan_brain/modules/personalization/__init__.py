"""Personalization module — implements ``interfaces.Personalizer``.

Responsibility: request-path adaptation. Select the traits relevant to *this* turn
(budget-aware, ~15% of context), inject them as system-prompt signals, tune tone/complexity/
proactive thresholds. Stateless — reads UserModel + FusedPerception, writes nothing.
Service: brain-api — `AdaptivePersonalizer` is built and wired into the request path; it injects
the compact profile + turn-relevant traits every turn (this is where learning becomes visible).

Files: adaptive.py (AdaptivePersonalizer).
"""
