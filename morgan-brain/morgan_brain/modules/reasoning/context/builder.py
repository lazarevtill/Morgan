"""Assemble the LLM message list from a ReasoningRequest: a system message carrying
personalization signals + the active skill + recalled memories, the prior history, then the
current user turn. Pure and deterministic.

Uses ``providers.wire.ChatMessage`` (the provider-neutral wire type) so messages can
be passed directly to any ``ChatClient`` adapter without conversion.
"""

from __future__ import annotations

from morgan_brain.interfaces.reasoning import ReasoningRequest
from morgan_brain.providers.wire import ChatMessage

_BASE_SYSTEM = (
    "You are Morgan, a personal assistant that knows the user well. "
    "Use the provided memories when relevant. If a memory conflicts with general knowledge, "
    "prefer the memory. Be helpful and concise."
)


def build_messages(request: ReasoningRequest) -> list[ChatMessage]:
    # Champion preprompt override is prepended before _BASE_SYSTEM so safety base stays.
    parts = []
    if request.system_override:
        parts.append(request.system_override)
    parts.append(_BASE_SYSTEM)
    if request.personalization.system_fragment:
        parts.append("About the user: " + request.personalization.system_fragment)
    if request.skill_prompt:
        parts.append("Active skill:\n" + request.skill_prompt)
    if request.memories:
        rendered = "\n".join(f"- {m.content}" for m in request.memories)
        parts.append("Relevant memories:\n" + rendered)

    messages: list[ChatMessage] = [ChatMessage(role="system", content="\n\n".join(parts))]
    for msg in request.history:
        # msg.role is a Role enum; .value gives the string ("user"/"assistant"/"system")
        messages.append(ChatMessage(role=msg.role.value, content=msg.content))
    messages.append(ChatMessage(role="user", content=request.perception.text))
    return messages
