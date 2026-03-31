"""
Skill loader.

Parses YAML frontmatter from markdown files and constructs Skill dataclass
instances. Supports loading single skills or scanning a directory of .md files.

Frontmatter fields:
    - name (str, required): Unique skill identifier.
    - description (str): Human-readable description.
    - when-to-use (str): Guidance on when the skill should be invoked.
    - allowed-tools (list[str] | str): Tools the skill's agent may use.
    - arguments (list[str] | str): Explicit argument names for the skill.
    - model (str): Optional model override.
    - effort (str): Effort level ("minimal", "balanced", "thorough").
    - agent (str): Optional agent type override.
    - user-invocable (bool): Whether users can invoke this skill directly.

The body of the .md file becomes the prompt template. ${var} patterns in the
body are replaced by the corresponding values from an args dict.

If no explicit 'arguments' field is present in the frontmatter, argument names
are auto-extracted from ${var} patterns found in the body.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from morgan.agents.loader import parse_frontmatter

logger = logging.getLogger(__name__)

# Pattern to match ${variable_name} in prompt templates
_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


def _parse_list_field(value: Any) -> List[str]:
    """
    Normalize a frontmatter field that can be a list or comma-separated string
    into a list of strings.

    Args:
        value: A list, a comma-separated string, or None.

    Returns:
        A list of stripped, non-empty strings.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _extract_variables(template: str) -> List[str]:
    """
    Extract unique ${var} variable names from a template string,
    preserving the order of first occurrence.

    Args:
        template: The prompt template body.

    Returns:
        Ordered list of unique variable names.
    """
    seen = set()
    result = []
    for match in _VAR_PATTERN.finditer(template):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


@dataclass
class Skill:
    """
    Defines a skill/plugin that can be executed through the agent system.

    Skills are loaded from markdown files with YAML frontmatter. The body
    of the file is a prompt template supporting ${var} substitution.

    Attributes:
        name: Unique skill identifier.
        description: Human-readable description of what this skill does.
        when_to_use: Guidance on when the skill should be invoked.
        allowed_tools: List of tool names the skill's agent may use.
        argument_names: List of argument names the skill accepts.
        model: Optional model override (None uses the default).
        effort: Effort level: "minimal", "balanced", or "thorough".
        agent: Optional agent type override.
        source: Origin of this skill (file path, "builtin", etc.).
        user_invocable: Whether users can invoke this skill directly.
        _content: The raw prompt template body with ${var} placeholders.
    """

    name: str
    description: str = ""
    when_to_use: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    argument_names: List[str] = field(default_factory=list)
    model: Optional[str] = None
    effort: str = "balanced"
    agent: Optional[str] = None
    source: str = "unknown"
    user_invocable: bool = True
    _content: str = ""

    def get_prompt(self, args: Optional[Dict[str, str]] = None) -> str:
        """
        Render the prompt template by substituting ${var} patterns with
        values from the provided args dict.

        Unknown variables (not present in args) are left as-is in the
        output.

        Args:
            args: Dictionary mapping variable names to their replacement
                  values. If None, returns the raw template.

        Returns:
            The rendered prompt string.
        """
        if args is None:
            return self._content

        result = self._content
        for key, value in args.items():
            result = result.replace(f"${{{key}}}", str(value))
        return result


def load_skill_from_file(path: str) -> Optional[Skill]:
    """
    Load a single Skill from a markdown file with YAML frontmatter.

    The file must have a 'name' field in its frontmatter. All other fields
    are optional. If no 'arguments' field is provided, argument names are
    auto-extracted from ${var} patterns in the body.

    Args:
        path: Path to the .md skill file.

    Returns:
        A Skill instance, or None if the file cannot be loaded.
    """
    file_path = Path(path)
    if not file_path.is_file():
        logger.warning("Skill file does not exist: %s", path)
        return None

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to read skill file %s: %s", path, e)
        return None

    meta, body = parse_frontmatter(content)

    # Name is the only required field
    name = meta.get("name")
    if not name:
        logger.warning(
            "Skipping %s: missing 'name' in frontmatter", path
        )
        return None

    # Parse allowed-tools (supports both hyphenated and underscored keys)
    allowed_tools = _parse_list_field(
        meta.get("allowed-tools") or meta.get("allowed_tools")
    )

    # Parse arguments
    explicit_args = _parse_list_field(
        meta.get("arguments") or meta.get("argument_names")
    )
    if explicit_args:
        argument_names = explicit_args
    else:
        # Auto-extract from template body
        argument_names = _extract_variables(body)

    return Skill(
        name=name,
        description=meta.get("description", ""),
        when_to_use=meta.get("when-to-use") or meta.get("when_to_use", ""),
        allowed_tools=allowed_tools,
        argument_names=argument_names,
        model=meta.get("model"),
        effort=meta.get("effort", "balanced"),
        agent=meta.get("agent"),
        source=str(file_path),
        user_invocable=meta.get("user-invocable", meta.get("user_invocable", True)),
        _content=body,
    )


def load_skills_from_dir(directory: str) -> List[Skill]:
    """
    Scan a directory for .md files and load each as a Skill.

    Files that fail to parse or lack a 'name' field are skipped with a
    warning log.

    Args:
        directory: Path to the directory containing .md skill files.

    Returns:
        List of Skill instances loaded from the directory.
        Returns an empty list if the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.debug("Skill directory does not exist: %s", directory)
        return []

    skills: List[Skill] = []

    for md_file in sorted(dir_path.glob("*.md")):
        skill = load_skill_from_file(str(md_file))
        if skill is not None:
            skills.append(skill)

    return skills
