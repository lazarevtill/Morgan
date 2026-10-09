"""Caller policy regressions use public MemoryGate and no model/network."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from examples.reader_policy_zero_model import (
    Event,
    ProofDesignation,
    ReaderAnswer,
    demonstrate,
    expected_answer,
    validate_reader,
)
from morgan_brain.composition import build_memory_context
from morgan_brain.config import Settings
from morgan_brain.models import Memory


async def inspection(
    tmp_path: Path, rows: list[tuple[str, str, list[int]]]
) -> tuple[dict, list[str]]:
    context = build_memory_context(
        Settings(
            data_dir=str(tmp_path),
            owner_user_id="owner",
            embedding_backend="hash",
            embedding_dim=16,
            _env_file=None,
        )
    )
    ids = []
    try:
        for index, (kind, value, parents) in enumerate(rows):
            ids.append(
                await context.gate.store(
                    Memory(
                        user_id="owner",
                        project="project",
                        source="tool_observed"
                        if kind == "proof"
                        else "agent_inferred"
                        if kind == "report"
                        else "user_stated",
                        author_id="tool:verifier" if kind == "proof" else "person:synthetic",
                        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index),
                        content=Event(task="task", kind=kind, value=value).model_dump_json(),
                        revises_event_ids=[ids[p] for p in parents],
                    )
                )
            )
        if not ids:
            ids = ["absent-source"]
        envelope = await context.gate.inspect_context(
            user_id="owner",
            project="project",
            evidence_ids=ids,
            selections=[],
            effective_at=datetime(2026, 1, 3, tzinfo=UTC),
        )
    finally:
        context.conn.close()
    return envelope, ids


def resolve(envelope: dict, proof: ProofDesignation | None = None) -> ReaderAnswer:
    return expected_answer(envelope, user_id="owner", project="project", task="task", proof=proof)


@pytest.mark.parametrize(
    "rows,status",
    [
        ([], "unknown"),
        ([("report", "self-asserted verified completed task", [])], "unverified_report"),
    ],
)
async def test_absence_and_assertion_differ(tmp_path: Path, rows: list, status: str) -> None:
    envelope, _ = await inspection(tmp_path, rows)
    result = resolve(envelope)
    assert result.verification_status == status and result.independently_verified is False
    assert result.completed_step is None and result.action_authority == "none"


@pytest.mark.parametrize(
    "rows,permission",
    [
        ([("permission", "granted", []), ("permission", "revoked", [0])], "revoked"),
        (
            [
                ("permission", "granted", []),
                ("permission", "revoked", [0]),
                ("permission", "granted", [1]),
            ],
            "granted",
        ),
        (
            [
                ("permission", "granted", []),
                ("permission", "granted", [0]),
                ("permission", "revoked", [0]),
            ],
            "unknown",
        ),
    ],
)
async def test_public_revision_permission_resolution(
    tmp_path: Path, rows: list, permission: str
) -> None:
    envelope, _ = await inspection(tmp_path, rows)
    assert resolve(envelope).shipping_permission == permission


async def test_artifact_proof_and_tamper(tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic-proof.json"
    artifact.write_text(
        json.dumps(
            {
                "user_id": "owner",
                "project": "project",
                "task": "task",
                "confirmed": True,
                "completed_step": "count-seven",
            }
        )
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    envelope, ids = await inspection(tmp_path / "db", [("proof", digest, [])])
    proof = ProofDesignation(ids[0], artifact, digest, "tool:verifier")
    assert resolve(envelope).verification_status == "unknown"  # undesignated proof is not proof
    verified = resolve(envelope, proof)
    assert verified.independently_verified and verified.completed_step == "count-seven"
    artifact.write_text("tampered")
    with pytest.raises(ValueError, match="designation mismatch"):
        resolve(envelope, proof)


async def test_wrong_artifact_scope_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "proof.json"
    artifact.write_text(
        json.dumps(
            {
                "user_id": "another-owner",
                "project": "project",
                "task": "task",
                "confirmed": True,
                "completed_step": None,
            }
        )
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    envelope, ids = await inspection(tmp_path / "db", [("proof", digest, [])])
    with pytest.raises(ValueError, match="artifact scope"):
        resolve(envelope, ProofDesignation(ids[0], artifact, digest, "tool:verifier"))


async def test_gate_wrong_project_hides_permission(tmp_path: Path) -> None:
    envelope, ids = await inspection(tmp_path, [("permission", "granted", [])])
    assert resolve(envelope).shipping_permission == "granted"
    context = build_memory_context(
        Settings(
            data_dir=str(tmp_path),
            owner_user_id="owner",
            embedding_backend="hash",
            embedding_dim=16,
            _env_file=None,
        )
    )
    try:
        wrong = await context.gate.inspect_context(
            user_id="owner", project="different-project", evidence_ids=ids, selections=[]
        )
        assert wrong["missing_ids"] == ids and wrong["sources"] == []
        assert (
            expected_answer(
                wrong, user_id="owner", project="different-project", task="task"
            ).shipping_permission
            == "unknown"
        )
    finally:
        context.conn.close()


async def test_raw_reader_rejected_without_rewriting(tmp_path: Path) -> None:
    envelope, _ = await inspection(
        tmp_path, [("report", "done", []), ("proposal", "next-proposal", [])]
    )
    expected = resolve(envelope)
    assert expected.proposed_step == "next-proposal" and expected.completed_step is None
    raw = expected.model_copy(update={"independently_verified": True}).model_dump_json()
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(raw)
    with pytest.raises(ValueError, match="reader answer conflicts"):
        validate_reader(raw, envelope, user_id="owner", project="project", task="task")
    assert raw_path.read_text() == raw


@pytest.mark.parametrize("value", [None, 0, "false"])
def test_strict_false_and_authority(value: object) -> None:
    with pytest.raises(ValueError):
        ReaderAnswer(
            verification_status="unknown",
            independently_verified=value,
            shipping_permission="unknown",
            proposed_step=None,
            completed_step=None,
            action_authority="none",
        )
    with pytest.raises(ValueError):
        ReaderAnswer(
            verification_status="unknown",
            independently_verified=False,
            shipping_permission="unknown",
            proposed_step=None,
            completed_step=None,
            action_authority="execute",
        )


async def test_executable_demo_preserves_raw_simulation(tmp_path: Path) -> None:
    result = await demonstrate(tmp_path)
    assert result["model_calls"] == result["actions_taken"] == 0 and result["simulation"]
    assert result["answer"]["shipping_permission"] == "revoked"
    raw = (tmp_path / "raw-simulated-reader-response.json").read_text()
    assert json.loads(raw) == result["answer"]


@pytest.mark.parametrize("confirmed", [False, 1])
async def test_unconfirmed_or_nonboolean_artifact_rejected(
    tmp_path: Path, confirmed: object
) -> None:
    artifact = tmp_path / "proof.json"
    artifact.write_text(
        json.dumps(
            {
                "user_id": "owner",
                "project": "project",
                "task": "task",
                "confirmed": confirmed,
                "completed_step": None,
            }
        )
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    envelope, ids = await inspection(tmp_path / "db", [("proof", digest, [])])
    with pytest.raises(ValueError):
        resolve(envelope, ProofDesignation(ids[0], artifact, digest, "tool:verifier"))


@pytest.mark.parametrize("excluded", ["semantic", "quarantined", "future"])
async def test_excluded_plaintext_does_not_affect_or_abort_policy(
    tmp_path: Path, excluded: str
) -> None:
    from copy import deepcopy

    envelope, _ = await inspection(tmp_path, [("permission", "revoked", [])])
    baseline = resolve(envelope)
    plain = deepcopy(envelope["sources"][0])
    plain["id"] = "excluded-plaintext"
    plain["content"] = "ordinary text; not an example typed JSON event"
    if excluded == "semantic":
        plain["kind"] = "semantic"
    elif excluded == "quarantined":
        plain["status"] = "quarantined"
    else:
        plain["created_at"] = "2027-01-01T00:00:00+00:00"
    envelope["sources"].insert(0, plain)
    assert resolve(envelope) == baseline


async def test_malformed_eligible_typed_event_fails_closed(tmp_path: Path) -> None:
    envelope, _ = await inspection(tmp_path, [("permission", "revoked", [])])
    envelope["sources"][0]["content"] = "ordinary text in an eligible typed-event input"
    with pytest.raises(ValueError):
        resolve(envelope)


async def offline_adapter(reply: str, finish_reason: str = "stop") -> tuple:
    import httpx
    import openai

    from morgan_brain.providers.openai_compat import OpenAICompatAdapter
    from morgan_brain.providers.request_budget import ChatRequestOptions

    captured = []

    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline-reader",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {"prompt_tokens": 23, "completion_tokens": 9, "total_tokens": 32},
            },
        )

    adapter = OpenAICompatAdapter(
        "http://offline.invalid/v1",
        "offline-test",
        "llamacpp",
        setting="OFFLINE_ENDPOINT",
        request_options=ChatRequestOptions(
            max_output_tokens=64, temperature=0.0, enable_thinking=False
        ),
    )
    await adapter._client.close()
    adapter._client = openai.AsyncOpenAI(
        base_url="http://offline.invalid/v1",
        api_key="offline-test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return adapter, captured


async def test_sdk_structured_policy_clean_path_preserves_raw_usage(tmp_path: Path) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    envelope, _ = await inspection(
        tmp_path / "db", [("report", "reported done", []), ("permission", "revoked", [])]
    )
    expected = resolve(envelope)
    raw = expected.model_dump_json()
    adapter, captured = await offline_adapter(raw)
    receipt = tmp_path / "raw-result.json"
    try:
        answer = await read_with_policy(
            adapter,
            [ChatMessage(role="user", content=json.dumps(envelope))],
            model="offline-model",
            raw_path=receipt,
            context=envelope,
            user_id="owner",
            project="project",
            task="task",
        )
        assert answer == expected and answer.action_authority == "none"
        assert len(captured) == 1 and captured[0]["max_tokens"] == 64
        assert captured[0]["temperature"] == 0.0
        assert captured[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert captured[0]["response_format"]["type"] == "json_schema"
        result = json.loads(receipt.read_text())
        assert result["text"] == raw and result["usage"] == {"input_tokens": 23, "output_tokens": 9}
    finally:
        await adapter._client.close()


@pytest.mark.parametrize("fault", ["schema", "forged-proof", "incomplete"])
async def test_sdk_bad_reader_fails_closed_preserving_raw(tmp_path: Path, fault: str) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    envelope, _ = await inspection(tmp_path / "db", [("report", "no proof", [])])
    answer = resolve(envelope).model_dump()
    if fault == "forged-proof":
        answer.update(verification_status="verified", independently_verified=True)
    elif fault == "schema":
        answer["independently_verified"] = "false"
    from morgan_brain.providers.structured import StructuredError

    raw = json.dumps(answer)
    error_type = StructuredError if fault == "schema" else ValueError
    adapter, captured = await offline_adapter(raw, "length" if fault == "incomplete" else "stop")
    receipt = tmp_path / "raw.json"
    try:
        with pytest.raises(error_type):
            await read_with_policy(
                adapter,
                [ChatMessage(role="user", content=json.dumps(envelope))],
                model="offline-model",
                raw_path=receipt,
                context=envelope,
                user_id="owner",
                project="project",
                task="task",
            )
        assert len(captured) == 1
        assert json.loads(receipt.read_text())["text"] == raw
    finally:
        await adapter._client.close()


async def test_sdk_stale_scope_refused_before_dispatch(tmp_path: Path) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    envelope, _ = await inspection(tmp_path / "db", [("report", "done", [])])
    adapter, captured = await offline_adapter(resolve(envelope).model_dump_json())
    receipt = tmp_path / "raw.json"
    try:
        with pytest.raises(ValueError, match="scope"):
            await read_with_policy(
                adapter,
                [ChatMessage(role="user", content="untrusted scope")],
                model="offline-model",
                raw_path=receipt,
                context=envelope,
                user_id="owner",
                project="stale-project",
                task="task",
            )
        assert captured == [] and not receipt.exists()
    finally:
        await adapter._client.close()


async def test_sdk_byte_guard_includes_forwarded_controls(tmp_path: Path) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.request_budget import StructuredRequestTooLarge
    from morgan_brain.providers.wire import ChatMessage

    envelope, _ = await inspection(tmp_path / "db", [("report", "done", [])])
    adapter, captured = await offline_adapter(resolve(envelope).model_dump_json())
    receipt = tmp_path / "raw.json"
    try:
        with pytest.raises(StructuredRequestTooLarge):
            await read_with_policy(
                adapter,
                [ChatMessage(role="user", content="source")],
                model="offline-model",
                raw_path=receipt,
                context=envelope,
                user_id="owner",
                project="project",
                task="task",
                request_byte_limit=1,
            )
        assert captured == [] and not receipt.exists()
    finally:
        await adapter._client.close()


async def test_sdk_designated_proof_clean_path(tmp_path: Path) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    artifact = tmp_path / "proof.json"
    artifact.write_text(
        json.dumps(
            {
                "user_id": "owner",
                "project": "project",
                "task": "task",
                "confirmed": True,
                "completed_step": "count-seven",
            }
        )
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    envelope, ids = await inspection(tmp_path / "db", [("proof", digest, [])])
    proof = ProofDesignation(ids[0], artifact, digest, "tool:verifier")
    expected = resolve(envelope, proof)
    adapter, captured = await offline_adapter(expected.model_dump_json())
    try:
        answer = await read_with_policy(
            adapter,
            [ChatMessage(role="user", content=json.dumps(envelope))],
            model="offline-model",
            raw_path=tmp_path / "raw.json",
            context=envelope,
            user_id="owner",
            project="project",
            task="task",
            proof=proof,
        )
        assert answer == expected and answer.action_authority == "none" and len(captured) == 1
    finally:
        await adapter._client.close()


async def test_sdk_stale_proof_refused_before_dispatch(tmp_path: Path) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    envelope, ids = await inspection(tmp_path / "db", [("proof", "invented-digest", [])])
    artifact = tmp_path / "proof.json"
    artifact.write_text("tampered artifact")
    adapter, captured = await offline_adapter("{}")
    try:
        with pytest.raises(ValueError, match="designation mismatch"):
            await read_with_policy(
                adapter,
                [ChatMessage(role="user", content="source")],
                model="offline-model",
                raw_path=tmp_path / "raw.json",
                context=envelope,
                user_id="owner",
                project="project",
                task="task",
                proof=ProofDesignation(ids[0], artifact, "invented-digest", "tool:verifier"),
            )
        assert not captured and not (tmp_path / "raw.json").exists()
    finally:
        await adapter._client.close()


@pytest.mark.parametrize("invalid", ["missing-parent", "existing-file", "directory"])
async def test_invalid_receipt_path_refused_before_sdk_call(tmp_path: Path, invalid: str) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    envelope, _ = await inspection(tmp_path / "db", [("report", "done", [])])
    adapter, captured = await offline_adapter(resolve(envelope).model_dump_json())
    receipt = (
        tmp_path / "missing-parent" / "raw.json"
        if invalid == "missing-parent"
        else tmp_path / "raw"
    )
    if invalid == "existing-file":
        receipt.write_text("prior response")
    elif invalid == "directory":
        receipt.mkdir()
    try:
        with pytest.raises((ValueError, OSError)):
            await read_with_policy(
                adapter,
                [ChatMessage(role="user", content="source")],
                model="offline-model",
                raw_path=receipt,
                context=envelope,
                user_id="owner",
                project="project",
                task="task",
            )
        assert captured == []
        if invalid == "existing-file":
            assert receipt.read_text() == "prior response"
    finally:
        await adapter._client.close()


@pytest.mark.parametrize("failure", ["write", "flush", "close"])
async def test_postresponse_receipt_failure_preserves_raw_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from examples.reader_policy_zero_model import ReaderReceiptError, read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    envelope, _ = await inspection(tmp_path / "db", [("report", "done", [])])
    adapter, captured = await offline_adapter(resolve(envelope).model_dump_json())
    receipt = tmp_path / "raw.json"
    real_open = Path.open
    closed = []

    class FailingOutput:
        def __init__(self, handle):
            self.handle = handle

        def write(self, value):
            if failure == "write":
                raise OSError("mock write failure")
            return self.handle.write(value)

        def flush(self):
            if failure == "flush":
                raise OSError("mock flush failure")
            self.handle.flush()

        def close(self):
            self.handle.close()
            closed.append(True)
            if failure == "close":
                raise OSError("mock close failure")

    def patched_open(path, mode="r", *args, **kwargs):
        handle = real_open(path, mode, *args, **kwargs)
        return FailingOutput(handle) if path == receipt and mode == "x" else handle

    monkeypatch.setattr(Path, "open", patched_open)
    try:
        with pytest.raises(ReaderReceiptError) as caught:
            await read_with_policy(
                adapter,
                [ChatMessage(role="user", content="source")],
                model="offline-model",
                raw_path=receipt,
                context=envelope,
                user_id="owner",
                project="project",
                task="task",
            )
        assert len(captured) == 1 and closed == [True]
        assert caught.value.raw_result.text == resolve(envelope).model_dump_json()
        assert caught.value.raw_result.usage.input_tokens == 23
        assert caught.value.raw_result.usage.output_tokens == 9
        assert caught.value.raw_result.finish_reason == "stop"
    finally:
        await adapter._client.close()


@pytest.mark.parametrize("invalid", ["missing_required", "unknown_option"])
async def test_sdk_reader_closed_keywords_fail_before_dispatch(
    tmp_path: Path, invalid: str
) -> None:
    from examples.reader_policy_zero_model import read_with_policy
    from morgan_brain.providers.wire import ChatMessage

    adapter, captured = await offline_adapter("unused")
    receipt = tmp_path / "raw-result.json"
    options = {
        "model": "offline-model",
        "raw_path": receipt,
        "context": {},
        "user_id": "owner",
        "project": "project",
        "task": "task",
    }
    if invalid == "missing_required":
        del options["task"]
    else:
        options["unexpected"] = "rejected"
    try:
        with pytest.raises(TypeError, match="invalid reader options"):
            await read_with_policy(
                adapter, [ChatMessage(role="user", content="synthetic")], **options
            )
        assert captured == [] and not receipt.exists()
    finally:
        await adapter._client.close()
