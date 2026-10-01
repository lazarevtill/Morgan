"""Execute frozen RU/EN operations against production, then compare declared gold."""

from pathlib import Path

import pytest

from morgan_brain.memory.revisions import RevisionError
from tests.revision_contract.production_adapter import error_reason, observe, operation_only
from tests.revision_contract.score_gold import declared_gold, load_fixture, score

FIXTURE = Path(__file__).resolve().parents[2] / "revision_contract" / "revision_acceptance_v1.json"
DATA = load_fixture(FIXTURE)


@pytest.mark.parametrize("scenario", DATA["scenarios"], ids=lambda row: row["id"])
async def test_production_matches_frozen_contract(scenario):
    actual = await observe(operation_only(scenario))
    actual["version"] = DATA["version"]
    result = score(declared_gold({"version": DATA["version"], "scenarios": [scenario]}), actual)
    assert result["all_passed"], result["failures"]


async def test_observer_refuses_gold_including_nested_candidate():
    with pytest.raises(ValueError, match="Gold"):
        await observe({"events": [{"expected": {"leaf_ids": ["gold"]}}]})


@pytest.mark.parametrize("error", [RevisionError("invented_reason"), RuntimeError("unknown")])
def test_unknown_error_cannot_be_mapped_using_gold(error):
    with pytest.raises(ValueError, match="Unrecognized"):
        error_reason(error)


def test_scorer_rejects_missing_result_and_forged_leaf():
    scenario = DATA["scenarios"][0]
    gold = declared_gold({"version": DATA["version"], "scenarios": [scenario]})
    assert not score(gold, {"version": DATA["version"], "observations": [], "writes": []})[
        "all_passed"
    ]
    import copy

    forged = copy.deepcopy(gold)
    forged["observations"][0]["eligible_leaf_ids"] = ["invented"]
    assert not score(gold, forged)["all_passed"]
