"""Which .env files a surface reads is a property of the surface, not of the process.

The CLI is run by the owner in a folder they chose, so ./.env is theirs. morgan-mcp is started
by a client in whatever folder that client happens to have open.
"""

from __future__ import annotations

import pytest

from morgan_brain.config import settings_for

#: These tests are about the real list of files each surface reads, so the root conftest's
#: guard (every other test reads none) stands aside. Each one runs in its own working
#: directory under the suite's empty XDG_CONFIG_HOME, so no developer file is in reach.
pytestmark = pytest.mark.reads_env_files


def test_a_cwd_env_moves_the_cli_and_not_the_mcp_server(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MORGAN_LLM_MODEL=from-cwd\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert settings_for("cli").llm_model == "from-cwd"
    assert settings_for("mcp").llm_model != "from-cwd"


def test_the_list_names_each_file_and_whether_it_was_there(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    files = settings_for("cli").env_files_read

    assert [f["path"] for f in files][-1].endswith(".env")
    assert files[-1]["present"] is False
    assert len(settings_for("mcp").env_files_read) == len(files) - 1
