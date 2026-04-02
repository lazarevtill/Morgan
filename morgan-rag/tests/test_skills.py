"""
Tests for the Skill/Plugin system.

Covers:
- Skill dataclass and get_prompt() template substitution
- load_skill_from_file() frontmatter parsing and argument extraction
- load_skills_from_dir() directory scanning
- SkillRegistry: register, get, list_all, list_user_invocable
- SkillExecutor: execute() creates AgentDefinition and delegates to AgentSpawner
- Bundled skills (web_research, code_review)
- Edge cases: missing fields, comma-separated lists, auto-extracted arguments
"""

import os
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional

import pytest

from morgan.skills.loader import (
    Skill,
    load_skill_from_file,
    load_skills_from_dir,
    _extract_variables,
    _parse_list_field,
)
from morgan.skills.registry import SkillRegistry
from morgan.skills.executor import SkillExecutor
from morgan.agents.base import AgentResult
from morgan.agents.spawner import AgentSpawner


# =============================================================================
# Helper Utilities Tests
# =============================================================================


class TestParseListField:
    """Tests for the _parse_list_field helper."""

    def test_none_returns_empty(self):
        assert _parse_list_field(None) == []

    def test_list_input(self):
        assert _parse_list_field(["a", "b", "c"]) == ["a", "b", "c"]

    def test_comma_separated_string(self):
        assert _parse_list_field("web_search, file_read, bash") == [
            "web_search",
            "file_read",
            "bash",
        ]

    def test_single_string(self):
        assert _parse_list_field("web_search") == ["web_search"]

    def test_empty_string(self):
        assert _parse_list_field("") == []

    def test_strips_whitespace(self):
        assert _parse_list_field(["  a  ", " b "]) == ["a", "b"]

    def test_filters_empty_items(self):
        assert _parse_list_field(["a", "", "b"]) == ["a", "b"]
        assert _parse_list_field("a,,b") == ["a", "b"]


class TestExtractVariables:
    """Tests for the _extract_variables helper."""

    def test_basic_extraction(self):
        assert _extract_variables("Hello ${name}, welcome to ${place}!") == [
            "name",
            "place",
        ]

    def test_no_variables(self):
        assert _extract_variables("No variables here.") == []

    def test_duplicate_variables(self):
        assert _extract_variables("${x} and ${y} and ${x} again") == [
            "x",
            "y",
        ]

    def test_preserves_order(self):
        assert _extract_variables("${c} ${a} ${b}") == ["c", "a", "b"]

    def test_underscore_in_name(self):
        assert _extract_variables("${file_path}") == ["file_path"]

    def test_ignores_malformed(self):
        assert _extract_variables("$notavar and ${valid}") == ["valid"]
        assert _extract_variables("${} empty") == []


# =============================================================================
# Skill Dataclass Tests
# =============================================================================


class TestSkill:
    """Tests for the Skill dataclass."""

    def test_basic_creation(self):
        skill = Skill(
            name="test_skill",
            description="A test skill",
            when_to_use="When testing",
            allowed_tools=["echo"],
            argument_names=["query"],
            _content="Search for ${query}",
        )
        assert skill.name == "test_skill"
        assert skill.description == "A test skill"
        assert skill.when_to_use == "When testing"
        assert skill.allowed_tools == ["echo"]
        assert skill.argument_names == ["query"]

    def test_defaults(self):
        skill = Skill(name="minimal")
        assert skill.description == ""
        assert skill.when_to_use == ""
        assert skill.allowed_tools == []
        assert skill.argument_names == []
        assert skill.model is None
        assert skill.effort == "balanced"
        assert skill.agent is None
        assert skill.source == "unknown"
        assert skill.user_invocable is True
        assert skill._content == ""

    def test_get_prompt_no_args(self):
        skill = Skill(
            name="test",
            _content="Hello ${name}, do ${task}.",
        )
        # None args returns raw template
        assert skill.get_prompt(None) == "Hello ${name}, do ${task}."
        # Empty args returns raw template
        assert skill.get_prompt({}) == "Hello ${name}, do ${task}."

    def test_get_prompt_with_args(self):
        skill = Skill(
            name="test",
            _content="Hello ${name}, do ${task}.",
        )
        result = skill.get_prompt({"name": "Morgan", "task": "research"})
        assert result == "Hello Morgan, do research."

    def test_get_prompt_partial_args(self):
        skill = Skill(
            name="test",
            _content="Hello ${name}, do ${task}.",
        )
        result = skill.get_prompt({"name": "Morgan"})
        assert result == "Hello Morgan, do ${task}."

    def test_get_prompt_extra_args_ignored(self):
        skill = Skill(
            name="test",
            _content="Hello ${name}.",
        )
        result = skill.get_prompt({"name": "Morgan", "unused": "value"})
        assert result == "Hello Morgan."

    def test_get_prompt_multiline(self):
        content = textwrap.dedent("""\
            Research the following topic:

            ## Query
            ${query}

            ## Instructions
            Be thorough.""")
        skill = Skill(name="test", _content=content)
        result = skill.get_prompt({"query": "AI safety"})
        assert "AI safety" in result
        assert "## Instructions" in result


# =============================================================================
# Skill Loader Tests
# =============================================================================


class TestLoadSkillFromFile:
    """Tests for load_skill_from_file."""

    def test_load_basic_skill(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: test_skill
                description: A test skill
                when-to-use: When testing
                allowed-tools:
                  - web_search
                  - file_read
                effort: thorough
                ---
                Research ${query} thoroughly.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.name == "test_skill"
                assert skill.description == "A test skill"
                assert skill.when_to_use == "When testing"
                assert skill.allowed_tools == ["web_search", "file_read"]
                assert skill.effort == "thorough"
                assert skill.argument_names == ["query"]
                assert "Research ${query}" in skill._content
                assert skill.source == f.name
            finally:
                os.unlink(f.name)

    def test_load_skill_comma_separated_tools(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: comma_skill
                allowed-tools: web_search, file_read, bash
                ---
                Do ${task}.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.allowed_tools == [
                    "web_search",
                    "file_read",
                    "bash",
                ]
            finally:
                os.unlink(f.name)

    def test_load_skill_explicit_arguments(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: explicit_args
                arguments:
                  - topic
                  - depth
                ---
                Research ${topic} to depth ${depth}.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.argument_names == ["topic", "depth"]
            finally:
                os.unlink(f.name)

    def test_load_skill_auto_extract_arguments(self):
        """When no arguments in frontmatter, extract from ${var} patterns."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: auto_args
                ---
                Review ${file_path} and report to ${user}.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.argument_names == ["file_path", "user"]
            finally:
                os.unlink(f.name)

    def test_load_skill_missing_name(self):
        """Skills without a name in frontmatter should return None."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                description: No name field
                ---
                Some content.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is None
            finally:
                os.unlink(f.name)

    def test_load_skill_no_frontmatter(self):
        """Files without frontmatter should return None (no name)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("Just plain text, no frontmatter.")
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is None
            finally:
                os.unlink(f.name)

    def test_load_skill_nonexistent_file(self):
        skill = load_skill_from_file("/nonexistent/path/to/skill.md")
        assert skill is None

    def test_load_skill_user_invocable_default(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: default_invocable
                ---
                Content.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.user_invocable is True
            finally:
                os.unlink(f.name)

    def test_load_skill_user_invocable_false(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: not_invocable
                user-invocable: false
                ---
                Internal skill.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.user_invocable is False
            finally:
                os.unlink(f.name)

    def test_load_skill_with_model_and_agent(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: custom_model
                model: gpt-4
                agent: researcher
                ---
                Research ${query}.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.model == "gpt-4"
                assert skill.agent == "researcher"
            finally:
                os.unlink(f.name)

    def test_load_skill_comma_separated_arguments(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(textwrap.dedent("""\
                ---
                name: comma_args
                arguments: topic, depth, format
                ---
                Research ${topic}.
            """))
            f.flush()

            try:
                skill = load_skill_from_file(f.name)
                assert skill is not None
                assert skill.argument_names == ["topic", "depth", "format"]
            finally:
                os.unlink(f.name)


class TestLoadSkillsFromDir:
    """Tests for load_skills_from_dir."""

    def test_load_multiple_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                f = Path(tmpdir) / f"skill_{i}.md"
                f.write_text(textwrap.dedent(f"""\
                    ---
                    name: skill_{i}
                    description: Skill number {i}
                    ---
                    Content for skill {i}.
                """))

            skills = load_skills_from_dir(tmpdir)
            assert len(skills) == 3
            names = {s.name for s in skills}
            assert names == {"skill_0", "skill_1", "skill_2"}

    def test_load_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills = load_skills_from_dir(tmpdir)
            assert skills == []

    def test_load_nonexistent_dir(self):
        skills = load_skills_from_dir("/nonexistent/path/to/skills")
        assert skills == []

    def test_load_skips_non_md_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md = Path(tmpdir) / "good.md"
            md.write_text(textwrap.dedent("""\
                ---
                name: good_skill
                ---
                Good skill content.
            """))
            txt = Path(tmpdir) / "bad.txt"
            txt.write_text("Not a skill.")

            skills = load_skills_from_dir(tmpdir)
            assert len(skills) == 1
            assert skills[0].name == "good_skill"

    def test_load_skips_invalid_skills(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid skill
            good = Path(tmpdir) / "good.md"
            good.write_text(textwrap.dedent("""\
                ---
                name: good_skill
                ---
                Good content.
            """))
            # Invalid skill (no name)
            bad = Path(tmpdir) / "bad.md"
            bad.write_text(textwrap.dedent("""\
                ---
                description: Missing name
                ---
                Bad content.
            """))

            skills = load_skills_from_dir(tmpdir)
            assert len(skills) == 1
            assert skills[0].name == "good_skill"


# =============================================================================
# SkillRegistry Tests
# =============================================================================


class TestSkillRegistry:
    """Tests for the SkillRegistry."""

    def test_register_and_get(self):
        registry = SkillRegistry()
        skill = Skill(name="test_skill", description="A test")
        registry.register(skill)

        retrieved = registry.get("test_skill")
        assert retrieved is not None
        assert retrieved.name == "test_skill"
        assert retrieved.description == "A test"

    def test_get_nonexistent(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_register_overwrites(self):
        registry = SkillRegistry()
        skill1 = Skill(name="dup", description="First", source="file1")
        skill2 = Skill(name="dup", description="Second", source="file2")

        registry.register(skill1)
        registry.register(skill2)

        retrieved = registry.get("dup")
        assert retrieved is not None
        assert retrieved.description == "Second"

    def test_list_all(self):
        registry = SkillRegistry()
        registry.register(Skill(name="bravo"))
        registry.register(Skill(name="alpha"))
        registry.register(Skill(name="charlie"))

        all_skills = registry.list_all()
        assert len(all_skills) == 3
        assert [s.name for s in all_skills] == ["alpha", "bravo", "charlie"]

    def test_list_all_empty(self):
        registry = SkillRegistry()
        assert registry.list_all() == []

    def test_list_user_invocable(self):
        registry = SkillRegistry()
        registry.register(
            Skill(name="public", user_invocable=True)
        )
        registry.register(
            Skill(name="internal", user_invocable=False)
        )
        registry.register(
            Skill(name="also_public", user_invocable=True)
        )

        user_skills = registry.list_user_invocable()
        assert len(user_skills) == 2
        names = [s.name for s in user_skills]
        assert "public" in names
        assert "also_public" in names
        assert "internal" not in names

    def test_list_user_invocable_sorted(self):
        registry = SkillRegistry()
        registry.register(Skill(name="zebra", user_invocable=True))
        registry.register(Skill(name="apple", user_invocable=True))

        user_skills = registry.list_user_invocable()
        assert [s.name for s in user_skills] == ["apple", "zebra"]


# =============================================================================
# SkillExecutor Tests
# =============================================================================


class TestSkillExecutor:
    """Tests for the SkillExecutor."""

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Execute a skill and verify the AgentDefinition is built correctly."""
        captured = {}

        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            captured["prompt"] = prompt
            captured["system_prompt"] = system_prompt
            captured["tools"] = tools
            captured["model"] = model
            return "Execution result"

        skill = Skill(
            name="test_exec",
            description="Test execution",
            when_to_use="Testing executor",
            allowed_tools=["web_search", "file_read"],
            argument_names=["query"],
            model="test-model",
            effort="thorough",
            _content="Research ${query} now.",
        )

        spawner = AgentSpawner(run_fn=mock_run)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(skill, {"query": "AI safety"})

        assert result.success is True
        assert result.output == "Execution result"
        assert result.agent_type == "skill:test_exec"
        assert captured["tools"] == ["web_search", "file_read"]
        assert captured["model"] == "test-model"
        # The prompt should have ${query} substituted
        assert "Research AI safety now." in captured["prompt"]

    @pytest.mark.asyncio
    async def test_execute_with_agent_override(self):
        """When skill has agent field, use it as agent_type."""

        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            return "ok"

        skill = Skill(
            name="custom_agent",
            agent="researcher",
            _content="Do research.",
        )

        spawner = AgentSpawner(run_fn=mock_run)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(skill)

        assert result.success is True
        assert result.agent_type == "researcher"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Context is passed through to the spawner."""
        captured = {}

        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            captured["prompt"] = prompt
            return "ok"

        skill = Skill(name="ctx_skill", _content="Do ${task}.")

        spawner = AgentSpawner(run_fn=mock_run)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(
            skill,
            args={"task": "analysis"},
            context="Previous conversation here",
        )

        assert result.success is True
        assert "Do analysis." in captured["prompt"]
        assert "Previous conversation here" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_execute_no_args(self):
        """Execute a skill with no arguments."""

        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            return prompt

        skill = Skill(
            name="no_args",
            _content="Just do the thing.",
        )

        spawner = AgentSpawner(run_fn=mock_run)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(skill)

        assert result.success is True
        assert "Just do the thing." in result.output

    @pytest.mark.asyncio
    async def test_execute_handles_error(self):
        """Executor propagates errors from the spawner."""

        async def failing_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            raise RuntimeError("LLM service down")

        skill = Skill(name="failing_skill", _content="Fail.")

        spawner = AgentSpawner(run_fn=failing_run)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(skill)

        assert result.success is False
        assert "LLM service down" in result.error

    @pytest.mark.asyncio
    async def test_execute_uses_when_to_use_over_description(self):
        """AgentDefinition.when_to_use prefers skill.when_to_use."""
        captured = {}

        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            return "ok"

        skill = Skill(
            name="pref_test",
            description="Description text",
            when_to_use="When-to-use text",
            _content="Content.",
        )

        spawner = AgentSpawner(run_fn=mock_run)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(skill)
        assert result.success is True


# =============================================================================
# Bundled Skills Tests
# =============================================================================


class TestBundledSkills:
    """Tests for the bundled skill markdown files."""

    BUNDLED_DIR = str(
        Path(__file__).parent.parent / "morgan" / "skills" / "bundled"
    )

    def test_bundled_dir_exists(self):
        assert Path(self.BUNDLED_DIR).is_dir()

    def test_load_bundled_skills(self):
        skills = load_skills_from_dir(self.BUNDLED_DIR)
        assert len(skills) >= 2
        names = {s.name for s in skills}
        assert "web_research" in names
        assert "code_review" in names

    def test_web_research_skill(self):
        skill = load_skill_from_file(
            str(Path(self.BUNDLED_DIR) / "web_research.md")
        )
        assert skill is not None
        assert skill.name == "web_research"
        assert "web_search" in skill.allowed_tools
        assert skill.effort == "thorough"
        assert skill.user_invocable is True
        assert "query" in skill.argument_names

        # Test template substitution
        prompt = skill.get_prompt({"query": "quantum computing"})
        assert "quantum computing" in prompt
        assert "${query}" not in prompt

    def test_code_review_skill(self):
        skill = load_skill_from_file(
            str(Path(self.BUNDLED_DIR) / "code_review.md")
        )
        assert skill is not None
        assert skill.name == "code_review"
        assert "file_read" in skill.allowed_tools
        assert skill.effort == "thorough"
        assert skill.user_invocable is True
        assert "file_path" in skill.argument_names

        # Test template substitution
        prompt = skill.get_prompt({"file_path": "/src/main.py"})
        assert "/src/main.py" in prompt
        assert "${file_path}" not in prompt


# =============================================================================
# Integration Tests
# =============================================================================


class TestSkillSystemIntegration:
    """End-to-end integration tests for the skill system."""

    BUNDLED_DIR = str(
        Path(__file__).parent.parent / "morgan" / "skills" / "bundled"
    )

    def test_load_register_and_retrieve(self):
        """Load bundled skills, register them, and retrieve by name."""
        skills = load_skills_from_dir(self.BUNDLED_DIR)
        registry = SkillRegistry()
        for skill in skills:
            registry.register(skill)

        web = registry.get("web_research")
        assert web is not None
        assert web.name == "web_research"

        code = registry.get("code_review")
        assert code is not None
        assert code.name == "code_review"

    @pytest.mark.asyncio
    async def test_load_and_execute_skill(self):
        """Load a bundled skill and execute it through the executor."""

        async def mock_run(
            prompt: str,
            system_prompt: str,
            tools: List[str],
            model: Optional[str],
        ) -> str:
            return f"Researched: {prompt[:50]}"

        skill = load_skill_from_file(
            str(Path(self.BUNDLED_DIR) / "web_research.md")
        )
        assert skill is not None

        spawner = AgentSpawner(run_fn=mock_run)
        executor = SkillExecutor(spawner=spawner)
        result = await executor.execute(
            skill, {"query": "latest AI breakthroughs"}
        )

        assert result.success is True
        assert "Researched:" in result.output
        assert result.agent_type == "skill:web_research"

    def test_registry_with_mixed_invocability(self):
        """Registry correctly filters user-invocable skills."""
        registry = SkillRegistry()

        # Load bundled (all user-invocable)
        skills = load_skills_from_dir(self.BUNDLED_DIR)
        for skill in skills:
            registry.register(skill)

        # Add a non-invocable skill
        registry.register(
            Skill(name="internal_tool", user_invocable=False)
        )

        all_skills = registry.list_all()
        user_skills = registry.list_user_invocable()

        assert len(all_skills) > len(user_skills)
        assert all(s.user_invocable for s in user_skills)
        assert not all(s.user_invocable for s in all_skills)

    @pytest.mark.asyncio
    async def test_full_pipeline_from_dir(self):
        """Full pipeline: dir -> load -> register -> get -> execute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a skill file
            skill_file = Path(tmpdir) / "summarize.md"
            skill_file.write_text(textwrap.dedent("""\
                ---
                name: summarize
                description: Summarize a document
                when-to-use: When user wants a summary
                allowed-tools:
                  - file_read
                effort: balanced
                ---
                Read and summarize the document at ${path}.
                Focus on: ${focus_area}
            """))

            # Load
            skills = load_skills_from_dir(tmpdir)
            assert len(skills) == 1

            # Register
            registry = SkillRegistry()
            for s in skills:
                registry.register(s)

            # Get
            skill = registry.get("summarize")
            assert skill is not None
            assert skill.argument_names == ["path", "focus_area"]

            # Execute
            async def mock_run(
                prompt: str,
                system_prompt: str,
                tools: List[str],
                model: Optional[str],
            ) -> str:
                assert "file_read" in tools
                return "Summary: The doc covers AI safety."

            spawner = AgentSpawner(run_fn=mock_run)
            executor = SkillExecutor(spawner=spawner)
            result = await executor.execute(
                skill,
                {"path": "/docs/paper.pdf", "focus_area": "key findings"},
            )

            assert result.success is True
            assert "Summary:" in result.output

    def test_import_from_package(self):
        """Verify public API is importable from morgan.skills."""
        from morgan.skills import (
            Skill,
            SkillRegistry,
            SkillExecutor,
            load_skill_from_file,
            load_skills_from_dir,
        )

        assert Skill is not None
        assert SkillRegistry is not None
        assert SkillExecutor is not None
        assert callable(load_skill_from_file)
        assert callable(load_skills_from_dir)
