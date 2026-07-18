"""
Tests for formatters.py and CLI --format / exit code behaviour.
"""

import json
import pytest
from click.testing import CliRunner
from unittest.mock import patch, AsyncMock

from promptlens.types import Phrase, SaliencyScore, SaliencyReport, RegionType
from promptlens_agent.formatters import (
    format_saliency_json,
    format_saliency_sarif,
    format_compression_json,
    format_audit_json,
    format_audit_sarif,
    _action_rationale,
)
from promptlens_agent.cli import cli


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_phrase(text: str, index: int = 0, char_start: int = 0, char_end: int = -1) -> Phrase:
    if char_end < 0:
        char_end = char_start + len(text)
    return Phrase(
        text=text,
        index=index,
        atomic=False,
        region_type=RegionType.PLAIN,
        char_start=char_start,
        char_end=char_end,
    )


def _make_score(text: str, index: int = 0, score: float = 0.05, char_start: int = 0) -> SaliencyScore:
    phrase = _make_phrase(text, index=index, char_start=char_start)
    return SaliencyScore(phrase=phrase, score=score, raw_shapley=score * 0.8, disposition="remove" if score < 0.15 else "keep")


def _make_report(scores: list[SaliencyScore], prompt: str = "You are a helpful assistant.") -> SaliencyReport:
    threshold = 0.15
    redundancy = sum(1 for s in scores if s.score < threshold) / len(scores) if scores else 0.0
    return SaliencyReport(
        prompt=prompt,
        phrases=[s.phrase for s in scores],
        scores=scores,
        token_count=len(prompt.split()),
        redundancy_fraction=redundancy,
        compression_candidate_tokens=sum(len(s.phrase.text.split()) for s in scores if s.score < threshold),
        m_samples=3,
        test_inputs_used=3,
        confidence=0.9,
    )


# ── format_saliency_json ─────────────────────────────────────────────────────


class TestSaliencyJson:
    def test_json_output_is_valid_json(self):
        scores = [_make_score("hello world", index=0, score=0.05)]
        output = format_saliency_json(scores, token_count=10, file="test.txt", threshold=0.15)
        parsed = json.loads(output)
        assert parsed["version"] == "1.0"
        assert parsed["command"] == "check"

    def test_json_contains_all_phrases(self):
        scores = [
            _make_score("phrase one", index=0, score=0.05, char_start=0),
            _make_score("phrase two", index=1, score=0.80, char_start=10),
            _make_score("phrase three", index=2, score=0.12, char_start=20),
        ]
        output = format_saliency_json(scores, token_count=15, file="prompt.txt", threshold=0.15)
        parsed = json.loads(output)
        assert parsed["phrase_count"] == 3
        assert len(parsed["phrases"]) == 3

    def test_json_redundancy_fraction(self):
        scores = [
            _make_score("low", index=0, score=0.05),
            _make_score("high", index=1, score=0.90),
        ]
        output = format_saliency_json(scores, token_count=10, file="f.txt", threshold=0.15)
        parsed = json.loads(output)
        assert parsed["redundancy_fraction"] == 0.5
        assert parsed["compression_candidate_count"] == 1

    def test_json_disposition_logic(self):
        # score < threshold * 0.5 => remove, score < threshold => compress, else keep
        threshold = 0.20
        scores = [
            _make_score("very low", index=0, score=0.05),   # < 0.10 => remove
            _make_score("medium low", index=1, score=0.15),  # < 0.20 => compress
            _make_score("high", index=2, score=0.80),        # >= 0.20 => keep
        ]
        output = format_saliency_json(scores, token_count=10, file="f.txt", threshold=threshold)
        parsed = json.loads(output)
        assert parsed["phrases"][0]["disposition"] == "remove"
        assert parsed["phrases"][1]["disposition"] == "compress"
        assert parsed["phrases"][2]["disposition"] == "keep"

    def test_json_includes_char_offsets(self):
        scores = [_make_score("hello", index=0, score=0.05, char_start=5)]
        output = format_saliency_json(scores, token_count=5, file="f.txt", threshold=0.15)
        parsed = json.loads(output)
        assert parsed["phrases"][0]["char_start"] == 5
        assert parsed["phrases"][0]["char_end"] == 10


# ── format_compression_json ───────────────────────────────────────────────────


class TestCompressionJson:
    def test_json_compression_includes_decisions(self):
        scores = [
            _make_score("remove this", index=0, score=0.03),
            _make_score("keep this", index=1, score=0.90),
        ]
        diff = [
            {"phrase": 0, "action": "remove", "original": "remove this", "result": ""},
            {"phrase": 1, "action": "keep", "original": "keep this", "result": "keep this"},
        ]
        output = format_compression_json(
            compressed_prompt="keep this",
            diff=diff,
            scores=scores,
            original_prompt="remove this keep this",
            verdict="PASS",
            worst_divergence=0.02,
            threshold=0.15,
        )
        parsed = json.loads(output)
        assert parsed["command"] == "compress"
        assert parsed["validation_verdict"] == "PASS"
        assert len(parsed["decisions"]) == 2
        # Each decision has a rationale
        for d in parsed["decisions"]:
            assert "rationale" in d
            assert len(d["rationale"]) > 0

    def test_json_compression_token_reduction(self):
        scores = [_make_score("a b c d", index=0, score=0.03)]
        diff = [{"phrase": 0, "action": "remove", "original": "a b c d", "result": ""}]
        output = format_compression_json(
            compressed_prompt="kept",
            diff=diff,
            scores=scores,
            original_prompt="a b c d kept",
            verdict="PASS",
            worst_divergence=0.01,
            threshold=0.15,
        )
        parsed = json.loads(output)
        assert parsed["original_tokens"] == 5
        assert parsed["compressed_tokens"] == 1
        assert parsed["token_reduction_pct"] == 80.0


# ── format_saliency_sarif ────────────────────────────────────────────────────


class TestSaliencySarif:
    def test_sarif_has_required_fields(self):
        scores = [_make_score("low phrase", index=0, score=0.05, char_start=0)]
        output = format_saliency_sarif(scores, token_count=5, file="test.txt", threshold=0.15)
        parsed = json.loads(output)
        assert parsed["$schema"].startswith("https://")
        assert parsed["version"] == "2.1.0"
        assert "runs" in parsed
        assert len(parsed["runs"]) == 1
        run = parsed["runs"][0]
        assert "tool" in run
        assert run["tool"]["driver"]["name"] == "PromptLens"
        assert "results" in run

    def test_sarif_locations_use_char_offsets(self):
        scores = [_make_score("test phrase", index=0, score=0.05, char_start=10)]
        output = format_saliency_sarif(scores, token_count=5, file="p.txt", threshold=0.15)
        parsed = json.loads(output)
        loc = parsed["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
        assert loc["region"]["charOffset"] == 10
        assert loc["region"]["charLength"] == 11  # len("test phrase")

    def test_sarif_only_reports_below_threshold(self):
        scores = [
            _make_score("low", index=0, score=0.05),
            _make_score("high", index=1, score=0.90),
        ]
        output = format_saliency_sarif(scores, token_count=10, file="f.txt", threshold=0.15)
        parsed = json.loads(output)
        results = parsed["runs"][0]["results"]
        # Only the low-saliency phrase should appear
        assert len(results) == 1
        assert "low" in results[0]["message"]["text"]

    def test_sarif_level_warning_for_very_low(self):
        scores = [_make_score("very low", index=0, score=0.03)]
        output = format_saliency_sarif(scores, token_count=5, file="f.txt", threshold=0.15)
        parsed = json.loads(output)
        assert parsed["runs"][0]["results"][0]["level"] == "warning"

    def test_sarif_level_note_for_moderate_low(self):
        scores = [_make_score("moderate", index=0, score=0.12)]
        output = format_saliency_sarif(scores, token_count=5, file="f.txt", threshold=0.15)
        parsed = json.loads(output)
        assert parsed["runs"][0]["results"][0]["level"] == "note"


# ── _action_rationale ─────────────────────────────────────────────────────────


class TestRationale:
    def test_rationale_varies_by_action(self):
        scores = [_make_score("text", index=0, score=0.05)]
        actions = ["keep", "remove", "rewrite", "merge", "paraphrase"]
        rationales = set()
        for action in actions:
            entry = {"phrase": 0, "action": action, "merge_target": 1}
            rationale = _action_rationale(entry, scores)
            rationales.add(rationale)
        # All actions should produce different rationale text
        assert len(rationales) == len(actions)

    def test_rationale_unknown_action(self):
        scores = [_make_score("text", index=0, score=0.50)]
        entry = {"phrase": 0, "action": "custom_action"}
        rationale = _action_rationale(entry, scores)
        assert "custom_action" in rationale


# ── format_audit_sarif ────────────────────────────────────────────────────────


class TestAuditSarif:
    def test_audit_sarif_multiple_files(self):
        all_scores = {
            "file1.txt": [_make_score("low1", index=0, score=0.05)],
            "file2.txt": [_make_score("low2", index=0, score=0.08)],
        }
        output = format_audit_sarif(all_scores, threshold=0.15)
        parsed = json.loads(output)
        results = parsed["runs"][0]["results"]
        assert len(results) == 2
        files_in_results = {r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"] for r in results}
        assert files_in_results == {"file1.txt", "file2.txt"}


# ── CLI exit code tests ───────────────────────────────────────────────────────


class TestCliExitCodes:
    @patch("promptlens_agent.cli.run_shapley", new_callable=AsyncMock)
    def test_exit_code_1_on_high_redundancy(self, mock_shapley, tmp_path):
        """check exits 1 when redundancy > 20%."""
        # Create a temp prompt file
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are a helpful assistant that does things.")

        # All phrases below threshold => high redundancy
        scores = [
            _make_score("You are a helpful", index=0, score=0.03),
            _make_score("assistant that", index=1, score=0.04),
            _make_score("does things", index=2, score=0.02),
        ]
        report = _make_report(scores)
        mock_shapley.return_value = report

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--file", str(prompt_file), "--format", "json"])
        assert result.exit_code == 1

    @patch("promptlens_agent.cli.run_shapley", new_callable=AsyncMock)
    def test_exit_code_0_on_clean_prompt(self, mock_shapley, tmp_path):
        """check exits 0 when redundancy <= 20%."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are a helpful assistant.")

        # All phrases above threshold => no redundancy
        scores = [
            _make_score("You are a helpful", index=0, score=0.90),
            _make_score("assistant", index=1, score=0.85),
        ]
        report = _make_report(scores)
        mock_shapley.return_value = report

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--file", str(prompt_file), "--format", "json"])
        assert result.exit_code == 0

    @patch("promptlens_agent.cli.run_shapley", new_callable=AsyncMock)
    def test_json_output_to_stdout(self, mock_shapley, tmp_path):
        """check --format json writes valid JSON to stdout."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt content.")

        scores = [_make_score("Test prompt content", index=0, score=0.90)]
        report = _make_report(scores, prompt="Test prompt content.")
        mock_shapley.return_value = report

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--file", str(prompt_file), "--format", "json"])
        assert result.exit_code == 0
        # stdout should be valid JSON
        parsed = json.loads(result.output)
        assert parsed["command"] == "check"
        assert parsed["version"] == "1.0"

    @patch("promptlens_agent.cli.run_shapley", new_callable=AsyncMock)
    def test_sarif_output_to_stdout(self, mock_shapley, tmp_path):
        """check --format sarif writes valid SARIF to stdout."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Low saliency content here.")

        scores = [_make_score("Low saliency content here", index=0, score=0.05)]
        report = _make_report(scores, prompt="Low saliency content here.")
        mock_shapley.return_value = report

        runner = CliRunner()
        result = runner.invoke(cli, ["check", "--file", str(prompt_file), "--format", "sarif"])
        # exit_code 1 because redundancy > 20%
        assert result.exit_code == 1
        parsed = json.loads(result.output)
        assert parsed["version"] == "2.1.0"
        assert "$schema" in parsed
