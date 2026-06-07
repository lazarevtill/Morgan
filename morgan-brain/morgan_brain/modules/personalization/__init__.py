"""Personalization module — implements ``interfaces.Personalizer``.

Responsibility: request-path adaptation. Select the traits relevant to *this* turn
(budget-aware, ~15% of context), inject them as system-prompt signals, tune tone/complexity/
proactive thresholds. Stateless — reads UserModel + FusedPerception, writes nothing.
Service: brain-api. Phase: 2.

Planned files: context/{selector,injector,assembler}.py, adaptation/{tone,complexity}.py.
"""
