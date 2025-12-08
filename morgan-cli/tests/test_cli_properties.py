"""
Property-based tests for CLI command history.

Feature: client-server-separation
"""

import pytest
from hypothesis import given, strategies as st, settings

from morgan_cli.cli import CommandHistory, get_command_history


# ============================================================================
# Property 21: Command history
# Validates: Requirements 9.3
# ============================================================================

@settings(max_examples=100)
@given(commands=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=50))
def test_command_history_stores_commands(commands):
    """
    Property 21: Command history
    
    For any sequence of commands entered by the user, the TUI client should
    maintain a history that allows navigating to previous commands using
    arrow keys or similar navigation.
    
    This test verifies that:
    1. Commands are stored in history
    2. Commands can be retrieved in order
    3. Navigation works correctly (previous/next)
    4. Empty commands are not stored
    5. Duplicate consecutive commands are not stored
    """
    history = CommandHistory()
    
    # Add all commands to history
    for cmd in commands:
        history.add(cmd)
    
    # Get all stored commands
    stored = history.get_all()
    
    # Filter expected commands (remove empty/whitespace-only and consecutive duplicates)
    expected = []
    for cmd in commands:
        if cmd and cmd.strip() and (not expected or expected[-1] != cmd):
            expected.append(cmd)
    
    # Property 1: All non-empty, non-duplicate commands should be stored
    assert stored == expected, (
        f"Command history should store all non-empty, non-duplicate commands. "
        f"Expected {expected}, got {stored}"
    )
    
    # Property 2: Can navigate backwards through history
    if expected:
        # Reset position
        history.position = len(history.history)
        
        retrieved_backwards = []
        for _ in range(len(expected)):
            cmd = history.get_previous()
            if cmd is not None:
                retrieved_backwards.append(cmd)
        
        # Should get commands in reverse order
        assert retrieved_backwards == list(reversed(expected)), (
            f"Navigating backwards should retrieve commands in reverse order. "
            f"Expected {list(reversed(expected))}, got {retrieved_backwards}"
        )
    
    # Property 3: Can navigate forwards through history
    if expected:
        # Move to beginning
        history.position = 0
        
        retrieved_forwards = []
        for _ in range(len(expected)):
            cmd = history.get_next()
            if cmd is not None and cmd != "":
                retrieved_forwards.append(cmd)
        
        # Should get commands in forward order (excluding the empty string at end)
        assert retrieved_forwards == expected[1:], (
            f"Navigating forwards should retrieve commands in forward order. "
            f"Expected {expected[1:]}, got {retrieved_forwards}"
        )


@settings(max_examples=100)
@given(
    commands=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20),
    back_steps=st.integers(min_value=0, max_value=10),
    forward_steps=st.integers(min_value=0, max_value=10)
)
def test_command_history_navigation_consistency(commands, back_steps, forward_steps):
    """
    Property: Navigation consistency
    
    For any sequence of navigation operations (back/forward), the history
    should maintain consistency and never return invalid positions.
    """
    history = CommandHistory()
    
    # Add commands
    for cmd in commands:
        history.add(cmd)
    
    stored = history.get_all()
    
    if not stored:
        return  # Nothing to test with empty history
    
    # Navigate backwards
    for _ in range(back_steps):
        cmd = history.get_previous()
        # Should either get a command or None (at beginning)
        if cmd is not None:
            assert cmd in stored, f"Retrieved command '{cmd}' not in history"
    
    # Navigate forwards
    for _ in range(forward_steps):
        cmd = history.get_next()
        # Should either get a command, empty string (at end), or None
        if cmd is not None and cmd != "":
            assert cmd in stored, f"Retrieved command '{cmd}' not in history"


@settings(max_examples=100)
@given(commands=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=20))
def test_command_history_empty_commands_not_stored(commands):
    """
    Property: Empty commands are not stored
    
    For any sequence of commands including empty strings, empty commands
    should not be stored in history.
    """
    history = CommandHistory()
    
    # Add commands including empty ones
    all_commands = commands + ["", "  ", ""]
    for cmd in all_commands:
        history.add(cmd)
    
    stored = history.get_all()
    
    # No empty or whitespace-only commands should be stored
    for cmd in stored:
        assert cmd.strip() != "", "Empty commands should not be stored in history"


@settings(max_examples=100)
@given(
    command=st.text(min_size=1, max_size=100).filter(lambda x: x.strip() != ""),
    repeat_count=st.integers(min_value=2, max_value=10)
)
def test_command_history_consecutive_duplicates_not_stored(command, repeat_count):
    """
    Property: Consecutive duplicate commands are not stored
    
    For any command repeated consecutively, only one instance should be
    stored in history.
    """
    history = CommandHistory()
    
    # Add the same command multiple times
    for _ in range(repeat_count):
        history.add(command)
    
    stored = history.get_all()
    
    # Should only have one instance (only if command is non-whitespace)
    assert len(stored) == 1, (
        f"Consecutive duplicate commands should not be stored. "
        f"Added {repeat_count} times, but got {len(stored)} in history"
    )
    assert stored[0] == command


@settings(max_examples=100)
@given(commands=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=1500))
def test_command_history_respects_max_size(commands):
    """
    Property: History respects maximum size
    
    For any sequence of commands exceeding the maximum history size,
    the oldest commands should be removed to maintain the size limit.
    """
    max_size = 1000
    history = CommandHistory(max_size=max_size)
    
    # Add all commands
    for cmd in commands:
        history.add(cmd)
    
    stored = history.get_all()
    
    # Should not exceed max size
    assert len(stored) <= max_size, (
        f"History size should not exceed {max_size}, but got {len(stored)}"
    )
    
    # If we added more than max_size unique commands, should have the most recent ones
    unique_commands = []
    for cmd in commands:
        if cmd and (not unique_commands or unique_commands[-1] != cmd):
            unique_commands.append(cmd)
    
    if len(unique_commands) > max_size:
        # Should have the last max_size commands
        expected = unique_commands[-max_size:]
        assert stored == expected, (
            f"History should contain the most recent {max_size} commands"
        )


def test_global_command_history_singleton():
    """
    Test that get_command_history returns the same instance.
    
    This ensures command history is shared across the application.
    """
    history1 = get_command_history()
    history2 = get_command_history()
    
    assert history1 is history2, "get_command_history should return the same instance"
    
    # Add a command to one
    history1.add("test command")
    
    # Should be visible in the other
    assert "test command" in history2.get_all()


@settings(max_examples=100)
@given(commands=st.lists(st.text(min_size=1, max_size=100).filter(lambda x: x.strip() != ""), min_size=1, max_size=20))
def test_command_history_position_reset_after_add(commands):
    """
    Property: Position resets after adding a command
    
    For any command added to history, the position should reset to the end,
    allowing the user to start navigating from the most recent command.
    """
    history = CommandHistory()
    
    # Add first command (guaranteed to be non-whitespace)
    history.add(commands[0])
    
    # Navigate backwards
    history.get_previous()
    
    # Position should not be at end
    assert history.position < len(history.history)
    
    # Add another command
    if len(commands) > 1:
        history.add(commands[1])
        
        # Position should be reset to end
        assert history.position == len(history.history), (
            "Position should reset to end after adding a command"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
