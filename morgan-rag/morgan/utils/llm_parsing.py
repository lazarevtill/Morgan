"""
LLM Response Parsing Utilities.

Provides robust parsing of LLM responses, handling various response formats
including thinking/reasoning blocks, markdown code blocks, and JSON extraction.
"""

import json
import re
from typing import Any, Dict, Optional

from .logger import get_logger

logger = get_logger(__name__)


def parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response, handling various formats.

    Handles:
    - Thinking/reasoning blocks (<think>...</think>)
    - Markdown code blocks (```json ... ```)
    - Plain JSON responses
    - JSON embedded in text

    Args:
        content: Raw LLM response content

    Returns:
        Parsed JSON as dict, or None if parsing fails
    """
    if not content or not content.strip():
        return None

    content = content.strip()

    try:
        # Handle thinking/reasoning blocks (from reasoning models like qwen3, deepseek-r1)
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # Handle markdown code blocks
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    content = part[4:].strip()
                    break
                elif part.startswith("{"):
                    content = part
                    break

        # Find JSON object in content
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            content = content[json_start:json_end]

        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error: {e}")
        return None
    except Exception as e:
        logger.debug(f"Error parsing LLM JSON: {e}")
        return None


def extract_json_from_llm_response(
    content: str, required_fields: Optional[list] = None
) -> Optional[Dict[str, Any]]:
    """
    Extract and validate JSON from LLM response.

    Args:
        content: Raw LLM response content
        required_fields: Optional list of required field names

    Returns:
        Validated JSON dict, or None if invalid
    """
    result = parse_llm_json(content)

    if result is None:
        return None

    if required_fields:
        for field in required_fields:
            if field not in result:
                logger.debug(f"Missing required field: {field}")
                return None

    return result


def clean_llm_response(content: str) -> str:
    """
    Clean LLM response by removing reasoning blocks and extracting main content.

    Args:
        content: Raw LLM response content

    Returns:
        Cleaned content
    """
    if not content:
        return ""

    content = content.strip()

    # Remove thinking blocks
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>")[-1].strip()

    # Remove markdown code blocks but keep content
    if "```" in content:
        # Extract content from code blocks
        parts = content.split("```")
        cleaned_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Inside code block
                # Remove language identifier
                lines = part.strip().split("\n")
                if lines and lines[0].strip() in ["json", "python", "javascript", "text"]:
                    lines = lines[1:]
                cleaned_parts.append("\n".join(lines))
            else:
                cleaned_parts.append(part)
        content = "\n".join(cleaned_parts)

    return content.strip()
