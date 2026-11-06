"""Explicit ntfy publication with transient credentials and no delivery/read claims."""

from __future__ import annotations

import asyncio
import json
from contextlib import closing, suppress
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from morgan_brain.local_notes import NoteWorkspace
from morgan_brain.tasks import Task, TaskControl, TaskKey, run_task

ACTION = "owner-ntfy-v1"


class Push(BaseModel):
    """Exact reviewed destination and text; no remote attachments, actions or callbacks."""

    model_config = ConfigDict(extra="forbid", strict=True)
    server: str
    topic: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    text: str = Field(min_length=1, max_length=2000)

    def endpoint(self) -> str:
        """Require an HTTPS origin without credentials, query, fragment or path."""
        parsed = urlsplit(self.server)
        origin_ok = parsed.scheme == "https" and bool(parsed.hostname)
        credentials = parsed.username is not None or parsed.password is not None
        extras = parsed.path not in ("", "/") or bool(parsed.query) or bool(parsed.fragment)
        if not origin_ok or credentials or extras:
            raise ValueError("push server must be an explicit HTTPS origin")
        if not self.text.strip() or self.text != self.text.strip():
            raise ValueError("push text must be nonblank with no surrounding whitespace")
        if len(self.text.encode("utf-8")) > 4096:
            raise ValueError("push text exceeds ntfy message limit")
        return self.server.rstrip("/") + "/" + self.topic


class OwnerPush:
    """One HTTP attempt; success is server acceptance, never device delivery or reading."""

    def __init__(
        self,
        workspace: NoteWorkspace,
        token: SecretStr,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Hold a transient token; an injected transport is a trusted testing seam."""
        if not token.get_secret_value().strip():
            raise ValueError("an explicit transient token is required")
        self.workspace = workspace
        self.token = token
        self.transport = transport

    def _ready(self, control: TaskControl) -> None:
        """Never infer sending authority from a persisted task or remembered text."""
        if control.authorize is None:
            raise ValueError("owner push requires an explicit authorizer")
        self.workspace.check_stop()
        control.check_ready()

    async def _publish(self, push: Push) -> str:
        """Bound the response, suppress proxy inheritance and never follow a redirect."""
        try:
            return await self._request(push)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError):
            raise RuntimeError("push outcome is unknown; automatic retry is forbidden") from None

    async def _request(self, push: Push) -> str:
        """Validate a bounded provider acknowledgement against the exact outgoing message."""
        async with (
            httpx.AsyncClient(
                transport=self.transport, trust_env=False, follow_redirects=False, timeout=10
            ) as client,
            client.stream(
                "POST",
                push.endpoint(),
                content=push.text.encode(),
                headers={
                    "Authorization": "Bearer " + self.token.get_secret_value(),
                    "Content-Type": "text/plain; charset=utf-8",
                },
            ) as response,
        ):
            if response.status_code != 200:
                raise ValueError("push server did not confirm acceptance")
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 16384:
                    raise ValueError("push response exceeds limit")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise TypeError("invalid push acknowledgement")
        actual = (result.get("event"), result.get("topic"), result.get("message"))
        if actual != ("message", push.topic, push.text):
            raise ValueError("push acknowledgement does not match request")
        message_id = result.get("id")
        if not isinstance(message_id, str) or not 1 <= len(message_id) <= 128:
            raise ValueError("invalid push message ID")
        return json.dumps(
            {
                "server_accepted": True,
                "message_id": message_id,
                "device_delivered": None,
                "read": None,
            }
        )

    async def __call__(self, task: Task, control: TaskControl) -> str:
        """Poll STOP while the network operation is pending and fence lost receipts."""
        if task.action != ACTION:
            raise ValueError("unexpected push action")
        push = Push.model_validate_json(task.payload)
        push.endpoint()
        self._ready(control)
        pending = asyncio.create_task(self._publish(push))
        try:
            async with asyncio.timeout(12):
                while not pending.done():
                    self._ready(control)
                    await asyncio.wait({pending}, timeout=0.1)
                self._ready(control)
                return pending.result()
        finally:
            if not pending.done():
                pending.cancel()
            with suppress(asyncio.CancelledError, RuntimeError):
                await pending


async def send(workspace: NoteWorkspace, key: TaskKey, push: Push, adapter: OwnerPush) -> Task:
    """Explicit account invocation approves only this exact destination, scope and text."""
    push.endpoint()
    if adapter.workspace is not workspace:
        raise ValueError("adapter must share the invocation workspace")
    payload = push.model_dump_json()

    def authorize(task: Task) -> None:
        """Check the current STOP marker and exact invocation specification."""
        workspace.check_stop()
        if (task.key, task.action, task.payload) != (key, ACTION, payload):
            raise ValueError("push differs from the explicitly approved request")

    with workspace.locked(), closing(workspace.journal()) as journal:
        workspace.check_stop()
        task = journal.create(key, action=ACTION, payload=payload)
        if task.status != "pending":
            return task
        return await run_task(
            journal, key, handlers={ACTION: adapter}, timeout_seconds=15, authorize=authorize
        )
