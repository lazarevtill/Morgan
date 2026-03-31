"""Hook type definitions."""

from enum import Enum


class HookType(Enum):
    """Events that can be hooked into."""

    MESSAGE_INBOUND = "message_inbound"
    MESSAGE_REPLY = "message_reply"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_COMPACT = "pre_compact"
    POST_COMPACT = "post_compact"
    CONFIG_CHANGE = "config_change"
