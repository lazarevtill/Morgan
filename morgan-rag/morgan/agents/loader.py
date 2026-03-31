"""
Agent loader.

Parses YAML frontmatter from markdown files and constructs AgentDefinition
instances. Supports loading agents from a directory of .md files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from morgan.agents.base import AgentDefinition

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter and body from markdown content.

    Frontmatter is delimited by '---' lines at the start of the content.
    Returns a tuple of (metadata_dict, body_string).

    Args:
        content: The full markdown content with optional frontmatter.

    Returns:
        Tuple of (metadata dict, body string). If no frontmatter is found,
        metadata is an empty dict and body is the full content.
    """
    content = content.strip()

    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}, content

    frontmatter_str = content[3:end_idx].strip()
    body = content[end_idx + 3 :].strip()

    try:
        metadata = yaml.safe_load(frontmatter_str)
        if metadata is None:
            metadata = {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse YAML frontmatter: %s", e)
        metadata = {}

    return metadata, body


def load_agents_from_dir(directory: str) -> List[AgentDefinition]:
    """
    Load agent definitions from .md files in a directory.

    Each .md file should have YAML frontmatter with at least:
    - agent_type: str
    - when_to_use: str (or 'description' as alias)
    - tools: list[str]

    Optional frontmatter fields:
    - disallowed_tools: list[str]
    - model: str
    - effort: str
    - max_turns: int

    The body of the .md file becomes the system prompt.

    Args:
        directory: Path to the directory containing .md agent files.

    Returns:
        List of AgentDefinition instances loaded from the directory.
        Returns an empty list if the directory does not exist or has no .md files.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        logger.debug("Agent directory does not exist: %s", directory)
        return []

    agents: List[AgentDefinition] = []

    for md_file in sorted(dir_path.glob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(content)

            # Validate required fields
            agent_type = meta.get("agent_type")
            if not agent_type:
                logger.warning(
                    "Skipping %s: missing 'agent_type' in frontmatter",
                    md_file,
                )
                continue

            when_to_use = meta.get("when_to_use") or meta.get(
                "description", ""
            )
            if not when_to_use:
                logger.warning(
                    "Skipping %s: missing 'when_to_use' in frontmatter",
                    md_file,
                )
                continue

            tools = meta.get("tools", [])
            if tools is None:
                tools = []

            # Capture body in closure
            prompt_body = body

            def make_prompt(text: str = prompt_body) -> str:
                return text

            agent = AgentDefinition(
                agent_type=agent_type,
                when_to_use=when_to_use,
                get_system_prompt=make_prompt,
                tools=tools,
                disallowed_tools=meta.get("disallowed_tools", []) or [],
                model=meta.get("model"),
                effort=meta.get("effort", "balanced"),
                max_turns=meta.get("max_turns", 10),
                source=str(md_file),
            )
            agents.append(agent)

        except Exception as e:
            logger.warning("Failed to load agent from %s: %s", md_file, e)
            continue

    return agents
