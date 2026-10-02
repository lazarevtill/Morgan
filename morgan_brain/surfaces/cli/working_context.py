"""Human and agent adapters for the opt-in working-context proposal boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from morgan_brain.app.continuation import ContinuationRequest, resume_work
from morgan_brain.app.working_context import WorkingContextService
from morgan_brain.composition import build_app_context, build_memory_context, utcnow
from morgan_brain.config import Settings
from morgan_brain.memory.checkpoints import CheckpointContext
from morgan_brain.memory.working_context import WorkingContextPreview
from morgan_brain.models import PERSONAL_PROJECT, MemorySource

MAX_PROPOSAL_BYTES = 131072


def load_proposal(path: str) -> dict[str, Any]:
    """Bound bytes before parsing even if the file grows while being read."""
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_PROPOSAL_BYTES + 1)
    if len(raw) > MAX_PROPOSAL_BYTES:
        raise ValueError("Working proposal file exceeds 131072 bytes")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError("Working proposal requires a JSON object")
    return value


async def read_context(
    settings: Settings, context_id: str | None, project: str | None = None
) -> dict[str, Any]:
    """Discover current named work or read one view; fetch details through evidence."""
    resolved = project if project is not None else PERSONAL_PROJECT
    ctx = build_memory_context(settings)
    try:
        if context_id is not None:
            view = await ctx.gate.get_working_context(
                context_id, user_id=settings.owner_user_id, project=resolved
            )
            return {
                "context_id": context_id,
                "project": resolved,
                "project_defaulted": project is None,
                "view": view.model_dump(mode="json") if view else None,
            }
        listing = await ctx.gate.list_working_contexts(
            user_id=settings.owner_user_id, project=resolved
        )
        contexts = [
            {
                "context_id": item.context_id,
                "fact_id": item.view.fact_id,
                "eligibility": item.view.eligibility,
                "title": item.view.state.title if item.view.state else None,
            }
            for item in listing.items
        ]
        return {
            "project": resolved,
            "project_defaulted": project is None,
            "contexts": contexts,
            "truncated": listing.truncated,
        }
    finally:
        ctx.conn.close()


# Explicit proposal options mirror the public tool schema without another wrapper type.
async def propose_context(  # pylint: disable=too-many-arguments
    settings: Settings,
    context_id: str,
    event_ids: list[str],
    *,
    project: str | None = None,
    author_id: str = "",
    rebuild: bool = False,
) -> dict[str, Any]:
    """Generate a reviewable proposal; this never applies it."""
    resolved = project if project is not None else PERSONAL_PROJECT
    ctx = build_app_context(settings)
    try:
        service = WorkingContextService(
            gate=ctx.gate,
            client=ctx.client,
            model=settings.llm_model,
            json_mode=settings.llm_json_mode,
        )
        preview = await service.preview(
            context_id,
            context=CheckpointContext(
                user_id=settings.owner_user_id,
                project=resolved,
                author_id=author_id,
            ),
            event_ids=event_ids,
            rebuild=rebuild,
        )
        result = {
            "project": resolved,
            "project_defaulted": project is None,
            "proposal": preview.model_dump(mode="json"),
            "applied": False,
        }
        # Include the wrapper, pretty CLI formatting and Windows CRLF output.
        # Old selected source bodies accumulate even when the new model feed fits.
        if len(
            (json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            .replace("\n", "\r\n")
            .encode("utf-8")
        ) > (MAX_PROPOSAL_BYTES):
            raise ValueError("Working proposal exceeds 131072 bytes; select fewer sources")
        return result
    finally:
        try:
            await ctx.chat.aclose()
        finally:
            try:
                await ctx.client.aclose()
            finally:
                ctx.conn.close()


async def apply_context(
    settings: Settings, proposal: dict[str, Any], project: str | None = None
) -> dict[str, Any]:
    """Validate input ownership, current exact sources and CAS before the explicit write."""
    if not isinstance(proposal, dict):
        raise TypeError("Working proposal requires a JSON object")
    if len(json.dumps(proposal, ensure_ascii=False).encode("utf-8")) > MAX_PROPOSAL_BYTES:
        raise ValueError("Working proposal exceeds 131072 bytes")
    preview = WorkingContextPreview.model_validate(proposal.get("proposal", proposal))
    resolved = project if project is not None else PERSONAL_PROJECT
    if (preview.context.user_id, preview.context.project) != (settings.owner_user_id, resolved):
        raise PermissionError("Proposal ownership/project must match the current caller")
    ctx = build_memory_context(settings)
    try:
        identity = await ctx.gate.put_working_context(preview)
        return {
            "context_id": preview.context_id,
            "fact_id": identity,
            "project": resolved,
            "project_defaulted": project is None,
            "applied": True,
        }
    finally:
        ctx.conn.close()


# Caller-reported identity stays explicit at the client boundary, separate from settings.
async def continue_context(  # pylint: disable=too-many-arguments
    settings: Settings,
    context_id: str,
    text: str,
    session_id: str,
    *,
    project: str | None = None,
    client: str = "cli",
    source: MemorySource = MemorySource.UNKNOWN,
    author_id: str = "",
) -> dict[str, Any]:
    """Create a useful draft in a fresh independent session; store both turn halves."""
    resolved = project if project is not None else PERSONAL_PROJECT
    ctx = build_app_context(settings)
    try:
        result = await resume_work(
            gate=ctx.gate,
            history=ctx.history,
            client=ctx.client,
            clock=utcnow,
            request=ContinuationRequest(
                model=settings.llm_model,
                context_id=context_id,
                user_id=settings.owner_user_id,
                project=resolved,
                text=text,
                session_id=session_id,
                caller_client=client,
                source=source,
                author_id=author_id,
            ),
        )
        return {
            **result.model_dump(mode="json"),
            "project": resolved,
            "project_defaulted": project is None,
        }
    finally:
        try:
            await ctx.chat.aclose()
        finally:
            try:
                await ctx.client.aclose()
            finally:
                ctx.conn.close()


async def cmd_context(  # pylint: disable=unused-argument
    args: argparse.Namespace, settings: Settings, project: str
) -> dict[str, Any]:
    # This personal-memory verb deliberately does not infer a project from a repository.
    named = args.project
    if args.context_action in ("show", "list"):
        return await read_context(settings, getattr(args, "context_id", None), named)
    if args.context_action == "propose":
        return await propose_context(
            settings,
            args.context_id,
            args.event_ids,
            project=named,
            author_id=args.author_id,
            rebuild=args.rebuild,
        )
    if args.context_action == "apply":
        proposal = await asyncio.to_thread(load_proposal, args.proposal_file)
        return await apply_context(settings, proposal, named)
    return await continue_context(
        settings,
        args.context_id,
        args.text,
        args.session_id,
        project=named,
        source=MemorySource(args.source),
        author_id=args.author_id,
    )


def _command(*arguments: str) -> str:
    """Display copyable literal arguments for the host shell; never execute them."""
    if os.name == "nt":
        return " ".join(
            value
            if re.fullmatch(r"[A-Za-z0-9_./:-]+", value)
            else "'" + value.replace("'", "''") + "'"
            for value in arguments
        )
    return shlex.join(arguments)


def render_context(data: dict[str, Any]) -> str:
    if "response" in data:
        command = _command(
            "morgan", "evidence", "--project=" + data["project"], "--", *data["source_event_ids"]
        )
        return (
            str(data["response"])
            + f"\n\nWorking context: {data['context_id']} ({data['project']})."
            + f"\nSource basis: {command}"
        )
    if "proposal" in data:
        return json.dumps(data, ensure_ascii=False, indent=2)
    if data.get("applied"):
        return f"Updated {data['context_id']} in {data['project']} ({data['fact_id']})."
    if "contexts" in data:
        rendered = (
            "\n".join(
                f"{item['context_id']}: {item['title'] or 'rebuild required'} "
                f"[{item['eligibility']}]"
                for item in data["contexts"]
            )
            or "No working contexts yet."
        )
        if data.get("truncated"):
            rendered += "\nOnly the first 32 names are shown; read a known context by ID."
        return rendered
    view = data.get("view")
    if not view or view["eligibility"] != "current":
        status = view["eligibility"] if view else "missing"
        rebuild = ["--rebuild"] if view else []
        command = _command(
            "morgan",
            "context",
            "propose",
            *rebuild,
            "--event-id",
            "CURRENT_SOURCE_ID",
            "--project=" + data["project"],
            "--json",
            "--",
            data["context_id"],
        )
        return (
            f"Working context {data['context_id']}: {status}.\n"
            f"{command}\n"
            "Review the proposal, then apply it explicitly."
        )
    state = view["state"]
    lines = [state["title"], "Selected source quotations; classifications remain unverified."]
    for decision in state["decisions"]:
        lines.append("Decision: " + decision["choice"]["quote"])
        if decision["reason"]:
            lines.append("Reason: " + decision["reason"]["quote"])
    for label, field in [
        ("Question", "open_questions"),
        ("Intention", "intentions"),
        ("Reported progress", "progress"),
    ]:
        lines.extend(label + ": " + item["quote"] for item in state[field])
    ids = list(dict.fromkeys(item["event_id"] for item in view["sources"]))
    lines.append(
        "Details: " + _command("morgan", "evidence", "--project=" + data["project"], "--", *ids)
    )
    return "\n".join(lines)
