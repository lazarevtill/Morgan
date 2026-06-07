"""Reasoning module — implements ``interfaces.Reasoner``.

Responsibility: assemble the context window, route to an LLM (fast/strong), optionally
plan/reflect and call tools/MCP, then generate. A thin pipeline, not a god class.
Service: brain-api. Phase: 1.

Planned files: llm/{router,client,fallback}.py (port from legacy services/llm),
context/{builder,compactor}.py, planning/{planner,reflection}.py, response/generator.py.
"""
