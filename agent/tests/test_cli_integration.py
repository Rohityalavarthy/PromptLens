"""Integration tests for CLI commands using Click's CliRunner."""
import json
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

import pytest
from click.testing import CliRunner

from promptlens_agent.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def tmp_prompt_file(tmp_path):
    """Create a temporary prompt file."""
    p = tmp_path / "test_prompt.txt"
    p.write_text("You are a helpful assistant. Please respond concisely and accurately. Always be polite.")
    return str(p)


def _make_mock_report():
    """Create a mock report object."""
    mock_report = MagicMock()
    mock_report.scores = []
    mock_report.token_count = 150
    mock_report.redundancy_fraction = 0.1
    return mock_report


class TestDryRun:
    def test_dry_run_does_not_call_api(self, runner, tmp_prompt_file):
        """Mock generate, invoke compress with --dry-run, verify generate not called."""
        mock_shapley = AsyncMock()
        with patch("promptlens_agent.cli.run_shapley", mock_shapley):
            result = runner.invoke(cli, [
                "compress", "--file", tmp_prompt_file, "--dry-run"
            ])
            # run_shapley should NOT be called in dry-run mode
            mock_shapley.assert_not_called()
            assert result.exit_code == 0

    def test_dry_run_json_output(self, runner, tmp_prompt_file):
        """Dry-run with JSON format outputs valid JSON with expected fields."""
        result = runner.invoke(cli, [
            "compress", "--file", tmp_prompt_file, "--dry-run", "--format", "json"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["version"] == "1.0"
        assert data["command"] == "compress"
        assert data["mode"] == "dry-run"
        assert "token_count" in data
        assert "phrase_count" in data
        assert "shapley_estimate" in data
        assert "compression_estimate" in data
        assert "total_estimate" in data

    def test_dry_run_terminal_output(self, runner, tmp_prompt_file):
        """Dry-run with terminal format shows readable output."""
        result = runner.invoke(cli, [
            "compress", "--file", tmp_prompt_file, "--dry-run", "--format", "terminal"
        ])
        assert result.exit_code == 0
        assert "Dry-run estimate for:" in result.output
        assert "Shapley analysis:" in result.output
        assert "Total estimated cost:" in result.output


class TestCIMode:
    def test_ci_mode_outputs_json_to_stdout(self, runner, tmp_prompt_file):
        """Invoke check with --ci, capture stdout, verify valid JSON."""
        mock_report = _make_mock_report()

        with patch("promptlens_agent.cli.run_shapley", AsyncMock(return_value=mock_report)):
            with patch("promptlens_agent.cli.format_saliency_json", return_value='{"result": "ok"}'):
                result = runner.invoke(cli, [
                    "check", "--file", tmp_prompt_file, "--ci"
                ])
                # Should output JSON (format defaults to json in CI mode)
                output = result.output.strip()
                data = json.loads(output)
                assert "result" in data

    def test_ci_mode_no_format_defaults_to_json(self, runner, tmp_prompt_file):
        """Invoke with --ci but no --format, verify JSON output."""
        mock_report = _make_mock_report()

        with patch("promptlens_agent.cli.run_shapley", AsyncMock(return_value=mock_report)):
            with patch("promptlens_agent.cli.format_saliency_json", return_value='{"format": "json"}') as mock_fmt:
                result = runner.invoke(cli, [
                    "check", "--file", tmp_prompt_file, "--ci"
                ])
                # format_saliency_json should be called (not terminal or sarif)
                mock_fmt.assert_called_once()


class TestBatchMode:
    def test_batch_and_file_mutually_exclusive(self, runner):
        """Invoke with both --batch and --file, verify error."""
        result = runner.invoke(cli, [
            "compress", "--batch", "--file", "some_file.txt"
        ])
        assert result.exit_code == 2
        assert "Cannot use --batch and --file together" in result.output

    def test_batch_requires_file_or_batch(self, runner):
        """Invoke compress without --file or --batch, verify error."""
        result = runner.invoke(cli, [
            "compress"
        ])
        assert result.exit_code == 2
        assert "Either --file or --batch is required" in result.output

    def test_batch_mode_json_output(self, tmp_path):
        """Batch mode with JSON format outputs expected structure."""
        batch_runner = CliRunner()
        mock_discovery = MagicMock()
        mock_discovery.confidence = 0.9
        mock_discovery.prompt_text = "You are helpful."
        mock_discovery.file = str(tmp_path / "test.py")
        mock_discovery.origin_file = None
        mock_discovery.line = 5

        mock_report = _make_mock_report()

        with patch("promptlens_agent.cli.discover_prompts", return_value=[mock_discovery]):
            with patch("promptlens_agent.cli.run_shapley", AsyncMock(return_value=mock_report)):
                with patch("promptlens_agent.cli.compress_prompt", AsyncMock(return_value=("compressed", []))):
                    with patch("promptlens_agent.cli.validate_compression", AsyncMock(return_value=("PASS", 0.05, "compressed"))):
                        result = batch_runner.invoke(cli, [
                            "compress", "--batch", "--repo", str(tmp_path), "--format", "json"
                        ])
                        assert result.exit_code == 0
                        # Extract JSON from output (may contain stderr progress lines mixed in)
                        output = result.output
                        json_start = output.index("{")
                        data = json.loads(output[json_start:])
                        assert data["version"] == "1.0"
                        assert data["command"] == "compress"
                        assert data["mode"] == "batch"
                        assert "results" in data
                        assert "summary" in data
