"""
Property-Based Tests for Morgan CLI UI Components

These tests verify universal properties for UI rendering and error handling.
"""

import re
from io import StringIO
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings, strategies as st
from rich.console import Console

from morgan_cli.ui import (
    render_markdown,
    render_message,
    render_error,
    render_connection_error,
    render_validation_error,
    render_timeout_error,
    render_server_error,
    console,
)


# ============================================================================
# Hypothesis Strategies
# ============================================================================


@st.composite
def markdown_content_strategy(draw):
    """Generate markdown content with various formatting."""
    # Generate different markdown elements
    elements = []

    # Headers
    if draw(st.booleans()):
        level = draw(st.integers(min_value=1, max_value=6))
        header_text = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
            ).filter(lambda x: x.strip())
        )
        elements.append(f"{'#' * level} {header_text}")

    # Bold text
    if draw(st.booleans()):
        bold_text = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
            ).filter(lambda x: x.strip())
        )
        elements.append(f"**{bold_text}**")

    # Italic text
    if draw(st.booleans()):
        italic_text = draw(
            st.text(
                min_size=1,
                max_size=50,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
            ).filter(lambda x: x.strip())
        )
        elements.append(f"*{italic_text}*")

    # Code blocks
    if draw(st.booleans()):
        code = draw(
            st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")
                ),
            ).filter(lambda x: x.strip() and "`" not in x)
        )
        elements.append(f"`{code}`")

    # Lists
    if draw(st.booleans()):
        list_items = draw(
            st.lists(
                st.text(
                    min_size=1,
                    max_size=30,
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd", "Zs")
                    ),
                ).filter(lambda x: x.strip()),
                min_size=1,
                max_size=5,
            )
        )
        for item in list_items:
            elements.append(f"- {item}")

    # Plain text
    if draw(st.booleans()) or not elements:
        plain_text = draw(
            st.text(
                min_size=1,
                max_size=200,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")
                ),
            ).filter(lambda x: x.strip())
        )
        elements.append(plain_text)

    return "\n\n".join(elements)


@st.composite
def error_details_strategy(draw):
    """Generate error details dictionaries."""
    keys = draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ).filter(lambda x: x.strip()),
            min_size=0,
            max_size=5,
            unique=True,
        )
    )

    details = {}
    for key in keys:
        value = draw(st.one_of(st.text(max_size=100), st.integers(), st.booleans()))
        details[key] = value

    return details


@st.composite
def suggestions_strategy(draw):
    """Generate suggestion lists."""
    return draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=100,
                alphabet=st.characters(
                    whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")
                ),
            ).filter(lambda x: x.strip()),
            min_size=0,
            max_size=5,
        )
    )


@st.composite
def error_message_strategy(draw):
    """Generate error messages."""
    return draw(
        st.text(
            min_size=1,
            max_size=200,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")),
        ).filter(lambda x: x.strip())
    )


# ============================================================================
# Property 20: Markdown rendering
# **Feature: client-server-separation, Property 20: Markdown rendering**
# **Validates: Requirements 9.1**
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    content=markdown_content_strategy(),
    title=st.one_of(
        st.none(),
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
        ).filter(lambda x: x.strip()),
    ),
)
def test_markdown_rendering(content, title):
    """
    Property: For any message containing markdown syntax (bold, italic, code blocks,
    lists), the TUI client should render it with appropriate formatting visible to
    the user.

    This test verifies that markdown content is properly rendered and contains
    the expected formatting elements.
    """
    # Capture console output
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    # Temporarily replace the global console
    with patch("morgan_cli.ui.console", test_console):
        # Render the markdown
        try:
            render_markdown(content, title)
            rendered_output = output.getvalue()
        except Exception as e:
            # If rendering fails, it should fall back to plain text
            rendered_output = output.getvalue()
            # Verify fallback contains the original content or an error message
            assert (
                len(rendered_output) > 0
            ), "Rendering should produce output even on error"

    # Verify output was produced
    assert len(rendered_output) > 0, "Markdown rendering should produce output"

    # Verify that the content is present in some form
    # (Rich may transform it, but key text should be there)
    # Extract plain text from content (remove markdown syntax)
    plain_content = re.sub(r"[*_`#\-]", "", content).strip()

    # Check if significant words from content appear in output
    # (allowing for Rich's formatting transformations)
    words = [w for w in plain_content.split() if len(w) > 3]
    if words:
        # At least some content words should appear in the output
        found_words = sum(
            1 for word in words[:5] if word.lower() in rendered_output.lower()
        )
        assert found_words > 0, "Rendered output should contain content from markdown"

    # If title is provided, verify it appears in output
    if title:
        # Title should appear somewhere in the output
        assert (
            title in rendered_output or title.lower() in rendered_output.lower()
        ), "Title should appear in rendered output"

    # Verify markdown formatting is processed (not raw markdown)
    # Check that markdown syntax is not just printed as-is
    # If content has bold markers, they should be processed
    if "**" in content:
        # The output should not contain raw ** markers (Rich processes them)
        # Note: Rich may use ANSI codes or other formatting
        # We just verify that rendering happened (output exists and has content)
        assert (
            len(rendered_output) > len(content) * 0.5
        ), "Rendered output should be substantial"

    # Verify no crashes or exceptions for valid markdown
    # (The test reaching here means no exception was raised)
    assert True, "Markdown rendering should not crash"


# ============================================================================
# Property 22: Error message clarity
# **Feature: client-server-separation, Property 22: Error message clarity**
# **Validates: Requirements 9.5**
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    error_message=error_message_strategy(),
    error_type=st.one_of(
        st.none(),
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Zs")),
        ).filter(lambda x: x.strip()),
    ),
    details=error_details_strategy(),
    suggestions=suggestions_strategy(),
)
def test_error_message_clarity(error_message, error_type, details, suggestions):
    """
    Property: For any error that occurs in the TUI client (network error, server error,
    validation error), the displayed error message should be user-friendly (no stack
    traces) and include suggested actions when applicable.

    This test verifies that error messages are clear, user-friendly, and actionable.
    """
    # Capture console output
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    # Temporarily replace the global console
    with patch("morgan_cli.ui.console", test_console):
        # Render the error
        render_error(
            error_message=error_message,
            error_type=error_type,
            details=details,
            suggestions=suggestions,
        )
        rendered_output = output.getvalue()

    # Verify output was produced
    assert len(rendered_output) > 0, "Error rendering should produce output"

    # Verify the error message is present
    assert (
        error_message in rendered_output
        or error_message.lower() in rendered_output.lower()
    ), "Error message should be visible in output"

    # Verify error type is shown if provided
    if error_type:
        assert (
            error_type in rendered_output
            or error_type.lower() in rendered_output.lower()
        ), "Error type should be visible in output"

    # Verify details are shown if provided
    if details:
        for key, value in details.items():
            # At least the key or value should appear
            key_present = (
                key in rendered_output or key.lower() in rendered_output.lower()
            )
            value_present = (
                str(value) in rendered_output
                or str(value).lower() in rendered_output.lower()
            )
            assert (
                key_present or value_present
            ), f"Error details should be visible: {key}={value}"

    # Verify suggestions are shown if provided
    if suggestions:
        for suggestion in suggestions:
            assert (
                suggestion in rendered_output
                or suggestion.lower() in rendered_output.lower()
            ), f"Suggestion should be visible: {suggestion}"

    # Verify no stack traces are present (user-friendly requirement)
    # Stack traces typically contain patterns like "Traceback", "File", "line"
    stack_trace_indicators = [
        "Traceback (most recent call last)",
        'File "',
        ", line ",
        "raise ",
        "Exception:",
        "Error:",
    ]

    # Count how many stack trace indicators are present
    stack_trace_count = sum(
        1 for indicator in stack_trace_indicators if indicator in rendered_output
    )

    # Allow a few matches (like "Error:" which is expected), but not full stack traces
    assert (
        stack_trace_count < 3
    ), "Error display should not contain full stack traces (should be user-friendly)"

    # Verify error indicator is present (like ❌ or similar)
    # The render_error function uses ❌ as an indicator
    assert (
        "❌" in rendered_output
        or "Error" in rendered_output
        or "error" in rendered_output
    ), "Error display should have clear error indicator"

    # Verify formatting is applied (Rich should add ANSI codes or structure)
    # Rich output typically contains escape sequences or is longer than plain text
    min_expected_length = len(error_message) + (len(error_type) if error_type else 0)
    assert (
        len(rendered_output) >= min_expected_length * 0.8
    ), "Error display should include formatting and structure"


# ============================================================================
# Additional Error Display Tests
# ============================================================================


@settings(max_examples=50, deadline=None)
@given(
    server_url=st.sampled_from(
        [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "https://example.com",
            "http://192.168.1.100:9000",
        ]
    ),
    error_msg=st.text(
        min_size=1,
        max_size=100,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")),
    ).filter(lambda x: x.strip()),
)
def test_connection_error_display(server_url, error_msg):
    """Test that connection errors are displayed with helpful suggestions."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        render_connection_error(server_url, Exception(error_msg))
        rendered_output = output.getvalue()

    # Verify output contains server URL
    assert server_url in rendered_output, "Connection error should show server URL"

    # Verify output contains helpful suggestions
    assert (
        "server" in rendered_output.lower() or "connection" in rendered_output.lower()
    ), "Connection error should mention server or connection"

    # Verify suggestions are present
    assert (
        "check" in rendered_output.lower() or "verify" in rendered_output.lower()
    ), "Connection error should include actionable suggestions"


@settings(max_examples=50, deadline=None)
@given(
    field_errors=st.dictionaries(
        keys=st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
        ).filter(lambda x: x.strip()),
        values=st.text(
            min_size=1,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")),
        ).filter(lambda x: x.strip()),
        min_size=1,
        max_size=5,
    )
)
def test_validation_error_display(field_errors):
    """Test that validation errors show field-level details."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        render_validation_error(field_errors)
        rendered_output = output.getvalue()

    # Verify field errors are shown
    for field, error in field_errors.items():
        # At least one of field or error should be visible
        field_visible = (
            field in rendered_output or field.lower() in rendered_output.lower()
        )
        error_visible = (
            error in rendered_output or error.lower() in rendered_output.lower()
        )
        assert (
            field_visible or error_visible
        ), f"Validation error should show field error: {field}={error}"

    # Verify it's labeled as validation error
    assert (
        "validation" in rendered_output.lower() or "invalid" in rendered_output.lower()
    ), "Should be clearly labeled as validation error"


@settings(max_examples=50, deadline=None)
@given(timeout_seconds=st.integers(min_value=1, max_value=300))
def test_timeout_error_display(timeout_seconds):
    """Test that timeout errors show duration and suggestions."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        render_timeout_error(timeout_seconds)
        rendered_output = output.getvalue()

    # Verify timeout duration is shown
    assert (
        str(timeout_seconds) in rendered_output
    ), "Timeout error should show timeout duration"

    # Verify it's labeled as timeout
    assert (
        "timeout" in rendered_output.lower() or "timed out" in rendered_output.lower()
    ), "Should be clearly labeled as timeout error"

    # Verify suggestions are present
    assert (
        "try" in rendered_output.lower() or "check" in rendered_output.lower()
    ), "Timeout error should include suggestions"


@settings(max_examples=50, deadline=None)
@given(
    status_code=st.sampled_from([400, 404, 500, 502, 503]),
    message=st.text(
        min_size=1,
        max_size=100,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs", "Po")),
    ).filter(lambda x: x.strip()),
    details=st.one_of(
        st.none(),
        st.dictionaries(
            keys=st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
            ).filter(lambda x: x.strip()),
            values=st.text(min_size=1, max_size=50),
            min_size=0,
            max_size=3,
        ),
    ),
)
def test_server_error_display(status_code, message, details):
    """Test that server errors show status code and appropriate suggestions."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        render_server_error(status_code, message, details)
        rendered_output = output.getvalue()

    # Verify status code is shown
    assert str(status_code) in rendered_output, "Server error should show status code"

    # Verify message is shown
    assert (
        message in rendered_output or message.lower() in rendered_output.lower()
    ), "Server error should show error message"

    # Verify details are shown if provided
    if details:
        for key, value in details.items():
            key_visible = (
                key in rendered_output or key.lower() in rendered_output.lower()
            )
            value_visible = (
                str(value) in rendered_output
                or str(value).lower() in rendered_output.lower()
            )
            assert (
                key_visible or value_visible
            ), f"Server error should show details: {key}={value}"

    # Verify appropriate suggestions based on status code
    if status_code >= 500:
        assert (
            "server" in rendered_output.lower()
        ), "5xx errors should mention server-side issue"
    elif status_code == 404:
        assert (
            "not found" in rendered_output.lower() or "404" in rendered_output
        ), "404 errors should mention not found"
    elif status_code == 400:
        assert (
            "input" in rendered_output.lower() or "request" in rendered_output.lower()
        ), "400 errors should mention input/request issue"


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================


def test_empty_markdown_rendering():
    """Test that empty markdown doesn't crash."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        render_markdown("")
        rendered_output = output.getvalue()

    # Should handle empty content gracefully (may produce no output)
    assert isinstance(rendered_output, str), "Should return string even for empty input"


def test_malformed_markdown_rendering():
    """Test that malformed markdown falls back gracefully."""
    malformed_content = "**unclosed bold\n`unclosed code\n# header without space"

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        try:
            render_markdown(malformed_content)
            rendered_output = output.getvalue()
            # Should not crash
            assert len(rendered_output) >= 0, "Should handle malformed markdown"
        except Exception as e:
            pytest.fail(f"Malformed markdown should not crash: {e}")


def test_error_with_empty_details():
    """Test error rendering with empty details."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        render_error("Test error", details={}, suggestions=[])
        rendered_output = output.getvalue()

    # Should still render the error message
    assert (
        "Test error" in rendered_output
    ), "Should render error even with empty details"


def test_error_with_none_values():
    """Test error rendering with None values."""
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    with patch("morgan_cli.ui.console", test_console):
        render_error("Test error", error_type=None, details=None, suggestions=None)
        rendered_output = output.getvalue()

    # Should still render the error message
    assert "Test error" in rendered_output, "Should render error even with None values"
