"""Reasoning module — implements ``interfaces.Reasoner``.

Responsibility: assemble the context window, route to an LLM via role (fast/strong) through the
provider seam, run the tool loop, then generate. A thin pipeline, not a god class.
Service: brain-api — built and wired as the request-path reason step. LLM routing/fallback/
structured output live behind the provider seam (`morgan_brain.providers`); tools/MCP are invoked
through the executor and PermissionGate.

Files: reasoner.py, context/builder.py.
"""
