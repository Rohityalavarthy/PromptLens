"""
Tests for compressor.py — covers the three pure functions:
  _build_compression_prompt, _parse_compression_response, reconstruct_from_original
No LLM calls are made.
"""
import pytest
from promptlens.types import Phrase, SaliencyScore, RegionType
from promptlens_agent.compressor import (
    _build_compression_prompt,
    _parse_compression_response,
    reconstruct_from_original,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_score(text: str, index: int = 0, score: float = 0.05) -> SaliencyScore:
    phrase = Phrase(text=text, index=index, region_type=RegionType.PLAIN)
    return SaliencyScore(phrase=phrase, score=score, raw_shapley=score, disposition="remove")


# ── _build_compression_prompt ────────────────────────────────────────────────

def test_prompt_contains_redundancy_pct():
    prompt = _build_compression_prompt(threshold=0.20, redundancy_pct=30)
    assert "30%" in prompt


def test_prompt_contains_threshold_as_boundary():
    prompt = _build_compression_prompt(threshold=0.20, redundancy_pct=20)
    assert "0.20" in prompt


def test_prompt_contains_all_four_actions():
    prompt = _build_compression_prompt(threshold=0.20, redundancy_pct=20)
    for action in ("REMOVE", "MERGE", "REWRITE", "PARAPHRASE"):
        assert action in prompt, f"{action} missing from prompt"


def test_prompt_single_template_regardless_of_threshold():
    low  = _build_compression_prompt(threshold=0.10, redundancy_pct=15)
    mid  = _build_compression_prompt(threshold=0.20, redundancy_pct=15)
    high = _build_compression_prompt(threshold=0.40, redundancy_pct=15)
    # All contain the same structural sections
    for p in (low, mid, high):
        assert "## Goal" in p
        assert "## Score-based decision guide" in p
        assert "## Output format" in p
        assert "## Hard rules" in p


def test_prompt_middle_row_omitted_when_threshold_at_or_below_0_10():
    prompt = _build_compression_prompt(threshold=0.10, redundancy_pct=10)
    # The middle band "0.10 – 0.10" should not appear
    assert "0.10 – 0.10" not in prompt


def test_prompt_middle_row_present_when_threshold_above_0_10():
    prompt = _build_compression_prompt(threshold=0.25, redundancy_pct=10)
    assert "0.10 – 0.25" in prompt


# ── _parse_compression_response ──────────────────────────────────────────────

def _scores(*texts):
    return [make_score(t, i) for i, t in enumerate(texts)]


def test_parse_keep():
    scores = _scores("You are helpful.")
    diff = _parse_compression_response("[KEEP] You are helpful.", scores)
    assert diff[0]["action"] == "keep"
    assert diff[0]["result"] == "You are helpful."


def test_parse_keep_robust_to_score_in_label():
    """LLM echoes [KEEP:0.94] instead of [KEEP] — should still parse correctly."""
    scores = _scores("You are helpful.")
    diff = _parse_compression_response("[KEEP:0.94] You are helpful.", scores)
    assert diff[0]["action"] == "keep"
    assert diff[0]["result"] == "You are helpful."


def test_parse_remove():
    scores = _scores("You are clear.")
    diff = _parse_compression_response("[REMOVE] You are clear.", scores)
    assert diff[0]["action"] == "remove"
    assert diff[0]["result"] == ""


def test_parse_merge():
    scores = _scores("You are helpful.", "You are friendly.")
    response = (
        "[KEEP] You are helpful.\n"
        "[MERGE with 1] You are friendly. → You are helpful and friendly."
    )
    diff = _parse_compression_response(response, scores)
    assert diff[1]["action"] == "merge"
    assert diff[1]["result"] == "You are helpful and friendly."


def test_parse_rewrite():
    scores = _scores("You are extremely detailed and thorough in your responses.")
    diff = _parse_compression_response(
        "[REWRITE] You are extremely detailed and thorough in your responses. → Be detailed and thorough.",
        scores,
    )
    assert diff[0]["action"] == "rewrite"
    assert diff[0]["result"] == "Be detailed and thorough."


def test_parse_paraphrase():
    scores = _scores("You assist users in a helpful manner.")
    diff = _parse_compression_response(
        "[PARAPHRASE] You assist users in a helpful manner. → Be helpful to users.",
        scores,
    )
    assert diff[0]["action"] == "paraphrase"
    assert diff[0]["result"] == "Be helpful to users."


def test_parse_empty_rewrite_becomes_remove():
    scores = _scores("You are clear.")
    diff = _parse_compression_response("[REWRITE] You are clear. → ", scores)
    assert diff[0]["action"] == "remove"
    assert diff[0]["result"] == ""


def test_parse_empty_merge_becomes_remove():
    scores = _scores("You are clear.")
    diff = _parse_compression_response("[MERGE with 1] You are clear. → ", scores)
    assert diff[0]["action"] == "remove"


def test_parse_empty_paraphrase_becomes_remove():
    scores = _scores("You are clear.")
    diff = _parse_compression_response("[PARAPHRASE] You are clear. → ", scores)
    assert diff[0]["action"] == "remove"


def test_parse_unparseable_line_falls_back_to_keep():
    scores = _scores("You are helpful.")
    diff = _parse_compression_response("some random line with no bracket", scores)
    assert diff[0]["action"] == "keep"
    assert diff[0]["original"] == "You are helpful."


def test_parse_multiline_response():
    scores = _scores("You are a chef.", "You are helpful.", "You are clear.")
    response = (
        "[KEEP] You are a chef.\n"
        "[PARAPHRASE] You are helpful. → Be helpful.\n"
        "[REMOVE] You are clear."
    )
    diff = _parse_compression_response(response, scores)
    assert diff[0]["action"] == "keep"
    assert diff[1]["action"] == "paraphrase"
    assert diff[1]["result"] == "Be helpful."
    assert diff[2]["action"] == "remove"


# ── reconstruct_from_original ────────────────────────────────────────────────

ORIGINAL = "You are a chef. You are helpful. You are clear."


def test_reconstruct_keep_is_noop():
    diff = [{"phrase": 0, "action": "keep", "original": "You are helpful.", "result": "You are helpful."}]
    assert reconstruct_from_original(ORIGINAL, diff) == ORIGINAL.strip()


def test_reconstruct_remove():
    diff = [{"phrase": 0, "action": "remove", "original": " You are clear.", "result": ""}]
    result = reconstruct_from_original(ORIGINAL, diff)
    assert "You are clear." not in result
    assert "You are a chef." in result


def test_reconstruct_rewrite():
    diff = [{"phrase": 0, "action": "rewrite", "original": "You are helpful.", "result": "Be helpful."}]
    result = reconstruct_from_original(ORIGINAL, diff)
    assert "Be helpful." in result
    assert "You are helpful." not in result


def test_reconstruct_paraphrase():
    diff = [{"phrase": 0, "action": "paraphrase", "original": "You are helpful.", "result": "Help users effectively."}]
    result = reconstruct_from_original(ORIGINAL, diff)
    assert "Help users effectively." in result
    assert "You are helpful." not in result


def test_reconstruct_cleans_multiple_blank_lines():
    original = "Line one.\n\n\n\nLine two."
    diff = [{"phrase": 0, "action": "remove", "original": "Line one.\n", "result": ""}]
    result = reconstruct_from_original(original, diff)
    assert "\n\n\n" not in result


def test_reconstruct_paraphrase_only_replaces_first_occurrence():
    original = "Be helpful. Be helpful."
    diff = [{"phrase": 0, "action": "paraphrase", "original": "Be helpful.", "result": "Assist users."}]
    result = reconstruct_from_original(original, diff)
    assert result.count("Assist users.") == 1
    assert result.count("Be helpful.") == 1
