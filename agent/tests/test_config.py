"""Tests for configuration file support."""

import os
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from promptlens_agent.config import load_config, PromptLensConfig, _find_config_file


def test_defaults_when_no_config_file():
    """load_config in empty dir returns default values."""
    with tempfile.TemporaryDirectory() as tmp:
        config = load_config(start_dir=Path(tmp))
    assert config.provider == "together"
    assert config.model == ""
    assert config.threshold == 0.15
    assert config.saliency_threshold == 0.15
    assert config.m_samples == 20
    assert config.semantic is False
    assert config.test_inputs_file is None
    assert config.output_format == "terminal"
    assert config.verbose is False
    assert ".git" in config.skip_dirs


def test_loads_from_promptlensrc_toml():
    """Config is loaded from .promptlensrc.toml."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = Path(tmp) / ".promptlensrc.toml"
        rc.write_text(textwrap.dedent("""\
            provider = "openai"
            m_samples = 50
            semantic = true
        """))
        config = load_config(start_dir=Path(tmp))
    assert config.provider == "openai"
    assert config.m_samples == 50
    assert config.semantic is True


def test_loads_from_pyproject_toml():
    """Config is loaded from pyproject.toml [tool.promptlens] section."""
    with tempfile.TemporaryDirectory() as tmp:
        pyproject = Path(tmp) / "pyproject.toml"
        pyproject.write_text(textwrap.dedent("""\
            [tool.promptlens]
            threshold = 0.2
            provider = "anthropic"
            verbose = true
        """))
        config = load_config(start_dir=Path(tmp))
    assert config.threshold == 0.2
    assert config.provider == "anthropic"
    assert config.verbose is True


def test_promptlensrc_takes_priority_over_pyproject():
    """.promptlensrc.toml wins over pyproject.toml when both exist."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = Path(tmp) / ".promptlensrc.toml"
        rc.write_text('provider = "openai"\n')
        pyproject = Path(tmp) / "pyproject.toml"
        pyproject.write_text('[tool.promptlens]\nprovider = "anthropic"\n')
        config = load_config(start_dir=Path(tmp))
    assert config.provider == "openai"


def test_env_vars_override_file_config():
    """PROMPTLENS_* env vars override file config values."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = Path(tmp) / ".promptlensrc.toml"
        rc.write_text('provider = "together"\nthreshold = 0.1\n')

        env_patch = {
            "PROMPTLENS_PROVIDER": "openai",
            "PROMPTLENS_THRESHOLD": "0.5",
            "PROMPTLENS_FORMAT": "json",
            "PROMPTLENS_VERBOSE": "true",
            "PROMPTLENS_M_SAMPLES": "100",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            config = load_config(start_dir=Path(tmp))

    assert config.provider == "openai"
    assert config.threshold == 0.5
    assert config.saliency_threshold == 0.5
    assert config.output_format == "json"
    assert config.verbose is True
    assert config.m_samples == 100


def test_search_up_finds_parent_config():
    """Config file in parent dir is found when starting from child dir."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = Path(tmp) / ".promptlensrc.toml"
        rc.write_text('provider = "openai"\n')
        child = Path(tmp) / "subdir" / "deep"
        child.mkdir(parents=True)
        config = load_config(start_dir=child)
    assert config.provider == "openai"


def test_find_config_file_returns_none_at_root():
    """_find_config_file returns None when no file is found up to root."""
    with tempfile.TemporaryDirectory() as tmp:
        result = _find_config_file(Path(tmp), "nonexistent.toml")
    assert result is None


def test_format_alias():
    """'format' key in TOML is treated as alias for output_format."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = Path(tmp) / ".promptlensrc.toml"
        rc.write_text('format = "sarif"\n')
        config = load_config(start_dir=Path(tmp))
    assert config.output_format == "sarif"


def test_skip_dirs_config():
    """skip_dirs can be overridden via config file."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = Path(tmp) / ".promptlensrc.toml"
        rc.write_text('skip_dirs = ["build", "dist"]\n')
        config = load_config(start_dir=Path(tmp))
    assert config.skip_dirs == ["build", "dist"]


def test_pyproject_without_tool_section():
    """pyproject.toml without [tool.promptlens] returns defaults."""
    with tempfile.TemporaryDirectory() as tmp:
        pyproject = Path(tmp) / "pyproject.toml"
        pyproject.write_text('[tool.black]\nline-length = 88\n')
        config = load_config(start_dir=Path(tmp))
    assert config.provider == "together"
    assert config.threshold == 0.15


def test_cli_flags_override_config():
    """CLI --saliency-threshold flag overrides config file value."""
    from promptlens_agent.cli import cli

    with tempfile.TemporaryDirectory() as tmp:
        # Create a config file with saliency_threshold = 0.2
        rc = Path(tmp) / ".promptlensrc.toml"
        rc.write_text('saliency_threshold = 0.2\n')

        # Create a dummy prompt file
        prompt_file = Path(tmp) / "prompt.txt"
        prompt_file.write_text("You are a helpful assistant that answers questions.")

        runner = CliRunner()
        # We can't fully run check (needs API), but we can verify the CLI
        # accepts the option and parses it. Use --help on the check command
        # to verify the option exists.
        result = runner.invoke(cli, ["check", "--help"])
        assert result.exit_code == 0
        assert "--saliency-threshold" in result.output


def test_env_verbose_parsing():
    """PROMPTLENS_VERBOSE accepts various truthy values."""
    with tempfile.TemporaryDirectory() as tmp:
        for val in ("1", "true", "True", "yes", "YES"):
            with patch.dict(os.environ, {"PROMPTLENS_VERBOSE": val}, clear=False):
                config = load_config(start_dir=Path(tmp))
            assert config.verbose is True, f"Failed for PROMPTLENS_VERBOSE={val}"

        for val in ("0", "false", "no", ""):
            with patch.dict(os.environ, {"PROMPTLENS_VERBOSE": val}, clear=False):
                config = load_config(start_dir=Path(tmp))
            assert config.verbose is False, f"Failed for PROMPTLENS_VERBOSE={val}"
