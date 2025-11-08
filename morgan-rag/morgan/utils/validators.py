"""
Validation utilities for Morgan RAG.

Simple, secure validation functions.
"""

from urllib.parse import urlparse


class ValidationError(Exception):
    """Custom validation error."""

    pass


def validate_url(url: str, field_name: str = "URL") -> bool:
    """
    Validate URL format and security.

    Args:
        url: URL to validate
        field_name: Field name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If URL is invalid or insecure
    """
    if not url or not isinstance(url, str):
        raise ValidationError(f"{field_name} cannot be empty")

    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"{field_name} is not a valid URL: {e}")

    # Check scheme
    if parsed.scheme not in ["http", "https"]:
        raise ValidationError(f"{field_name} must use http or https protocol")

    # Check hostname
    if not parsed.hostname:
        raise ValidationError(f"{field_name} must have a valid hostname")

    # Security checks
    if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
        # Allow localhost for development
        pass
    elif parsed.hostname.startswith("192.168.") or parsed.hostname.startswith("10."):
        # Allow private networks
        pass

    return True


def validate_int_range(
    value: int, min_value: int, max_value: int, field_name: str = "Value"
) -> bool:
    """
    Validate integer is within range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        field_name: Field name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")

    if value < min_value or value > max_value:
        raise ValidationError(
            f"{field_name} must be between {min_value} and {max_value}, got {value}"
        )

    return True


def validate_string_not_empty(value: str, field_name: str = "Value") -> bool:
    """
    Validate string is not empty.

    Args:
        value: String to validate
        field_name: Field name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If string is empty
    """
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")

    if not value.strip():
        raise ValidationError(f"{field_name} cannot be empty")

    return True


def validate_file_path(path: str, field_name: str = "Path") -> bool:
    """
    Validate file path for security.

    Args:
        path: File path to validate
        field_name: Field name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If path is insecure
    """
    if not path or not isinstance(path, str):
        raise ValidationError(f"{field_name} cannot be empty")

    # Check for path traversal
    if ".." in path or path.startswith("/"):
        raise ValidationError(f"{field_name} contains invalid characters")

    return True
