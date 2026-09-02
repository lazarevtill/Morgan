"""ACE-style playbook curation: incremental delta updates that preserve existing detail,
preventing the context-collapse / brevity-bias of iterative full-document rewrites.
"""

from __future__ import annotations

from morgan_brain.learning.optimizer import curate_playbook


def test_appends_new_bullets_and_preserves_existing() -> None:
    current = "- be concise\n- use code blocks for code"
    out = curate_playbook(current, ["prefer metric units"], char_budget=1000)
    # existing detail preserved verbatim, new bullet appended
    assert "be concise" in out
    assert "use code blocks for code" in out
    assert "prefer metric units" in out


def test_dedups_normalised_duplicates_idempotent() -> None:
    current = "- Be Concise"
    out = curate_playbook(current, ["be concise", "  - BE  CONCISE  "], char_budget=1000)
    # only one "be concise" bullet survives regardless of markers/case/spacing
    assert out.lower().count("be concise") == 1


def test_does_not_rewrite_or_summarise_existing_detail() -> None:
    # The whole point: a detailed existing bullet is kept verbatim, not compressed.
    detailed = "- never call the production database directly; always go through the service layer"
    out = curate_playbook(detailed, ["log every deploy"], char_budget=1000)
    assert detailed.lstrip("- ") in out


def test_budget_cap_drops_oldest_whole_bullets_keeps_newest() -> None:
    current = "\n".join(f"- old rule {i} with some length to it" for i in range(10))
    out = curate_playbook(current, ["the newest most important rule"], char_budget=120)
    assert len(out) <= 120
    # newest learning is retained; some oldest bullets are dropped wholesale (not summarised)
    assert "the newest most important rule" in out
    assert "old rule 0" not in out


def test_normalises_unbulleted_input_to_dashes() -> None:
    out = curate_playbook("plain line one", ["plain line two"], char_budget=1000)
    assert "- plain line one" in out
    assert "- plain line two" in out
