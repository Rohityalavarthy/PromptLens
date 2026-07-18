"""
Tests for compressor.py — covers the three pure functions:
  _build_compression_prompt, _parse_compression_response, reconstruct_from_original
No LLM calls are made.
"""
import pytest
import warnings
from promptlens.types import Phrase, SaliencyScore, RegionType
from promptlens_agent.compressor import (
    _build_compression_prompt,
    _parse_compression_response,
    _validate_coverage,
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


def test_parse_rewrite_longer_than_original_becomes_keep():
    scores = _scores("You are helpful.")
    # 3 words → 4 words: not a compression
    diff = _parse_compression_response(
        "[REWRITE] You are helpful. → You assist with inquiries.",
        scores,
    )
    assert diff[0]["action"] == "keep"
    assert diff[0]["result"] == "You are helpful."


def test_parse_paraphrase_longer_than_original_becomes_keep():
    scores = _scores("Be concise.")
    diff = _parse_compression_response(
        "[PARAPHRASE] Be concise. → Keep your responses brief and to the point.",
        scores,
    )
    assert diff[0]["action"] == "keep"


def test_parse_rewrite_same_length_becomes_keep():
    scores = _scores("You are helpful.")  # 3 words
    diff = _parse_compression_response(
        "[REWRITE] You are helpful. → You are useful.",  # also 3 words
        scores,
    )
    assert diff[0]["action"] == "keep"


def test_parse_rewrite_strictly_shorter_is_accepted():
    scores = _scores("You are extremely helpful.")  # 4 words
    diff = _parse_compression_response(
        "[REWRITE] You are extremely helpful. → Be helpful.",  # 2 words
        scores,
    )
    assert diff[0]["action"] == "rewrite"
    assert diff[0]["result"] == "Be helpful."


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


# ── New ID-based parser tests ────────────────────────────────────────────────

def test_parse_id_based_format():
    """[#0][KEEP] text parses correctly by ID."""
    scores = _scores("Always respond concisely.", "Be brief.", "Keep it short.")
    response = (
        "[#0][KEEP] Always respond concisely.\n"
        "[#1][REMOVE] Be brief.\n"
        "[#2][REWRITE] Keep it short. → Be short."
    )
    diff = _parse_compression_response(response, scores)
    assert any(d["phrase"] == 0 and d["action"] == "keep" for d in diff)
    assert any(d["phrase"] == 1 and d["action"] == "remove" for d in diff)
    assert any(d["phrase"] == 2 and d["action"] == "rewrite" and d["result"] == "Be short." for d in diff)


def test_parse_handles_skipped_phrase():
    """LLM outputs 4 lines for 5 phrases, missing one defaults to KEEP."""
    scores = _scores("A", "B", "C", "D", "E")
    response = (
        "[#0][KEEP] A\n"
        "[#1][REMOVE] B\n"
        "[#3][REMOVE] D\n"
        "[#4][KEEP] E"
    )
    diff = _parse_compression_response(response, scores)
    # Phrase #2 should default to KEEP
    phrase_2 = next(d for d in diff if d["phrase"] == 2)
    assert phrase_2["action"] == "keep"


def test_parse_handles_extra_lines():
    """LLM outputs 6 lines for 5 phrases, extras ignored."""
    scores = _scores("A", "B", "C", "D", "E")
    response = (
        "[#0][KEEP] A\n"
        "[#1][REMOVE] B\n"
        "[#2][KEEP] C\n"
        "[#3][KEEP] D\n"
        "[#4][KEEP] E\n"
        "[#99][KEEP] Extra line that should be ignored"
    )
    diff = _parse_compression_response(response, scores)
    # Should have exactly 5 entries (one per score)
    assert len(diff) == 5
    phrase_ids = {d["phrase"] for d in diff}
    assert phrase_ids == {0, 1, 2, 3, 4}


def test_parse_handles_duplicate_ids():
    """First occurrence wins for duplicate IDs."""
    scores = _scores("Hello.", "World.")
    response = (
        "[#0][KEEP] Hello.\n"
        "[#1][REMOVE] World.\n"
        "[#1][KEEP] World."  # duplicate — should be ignored
    )
    diff = _parse_compression_response(response, scores)
    phrase_1 = next(d for d in diff if d["phrase"] == 1)
    assert phrase_1["action"] == "remove"


def test_parse_fallback_to_content_match():
    """Line without [#N] matched by text comparison to unmatched scores."""
    scores = _scores("You are helpful.", "Be concise.")
    response = (
        "[#0][KEEP] You are helpful.\n"
        "Be concise."  # no bracket prefix — should be content-matched
    )
    diff = _parse_compression_response(response, scores)
    assert len(diff) == 2
    phrase_1 = next(d for d in diff if d["phrase"] == 1)
    assert phrase_1["action"] == "keep"


def test_merge_into_target_removes_source():
    """[#2][MERGE into #1] sets phrase 2 to remove (after merge semantics applied)."""
    scores = _scores("A", "B", "C")
    response = (
        "[#0][KEEP] A\n"
        "[#1][KEEP] B\n"
        "[#2][MERGE into #1] C"
    )
    from promptlens_agent.compressor import _apply_merge_semantics
    diff = _parse_compression_response(response, scores)
    diff = _apply_merge_semantics(diff, scores)
    phrase_2 = next(d for d in diff if d["phrase"] == 2)
    assert phrase_2["action"] == "merge"
    assert phrase_2["result"] == ""  # source is removed


def test_merge_invalid_target_becomes_remove():
    """Merge into removed phrase → warning + remove."""
    scores = _scores("A", "B", "C")
    response = (
        "[#0][KEEP] A\n"
        "[#1][REMOVE] B\n"
        "[#2][MERGE into #1] C"
    )
    from promptlens_agent.compressor import _apply_merge_semantics
    diff = _parse_compression_response(response, scores)
    diff = _apply_merge_semantics(diff, scores)
    phrase_2 = next(d for d in diff if d["phrase"] == 2)
    assert phrase_2["action"] == "remove"
    assert phrase_2["result"] == ""


def test_reconstruct_offset_based():
    """Prompt with duplicate phrases — only the correct one is modified using offsets."""
    original = "Be helpful. Be concise. Be helpful."
    # Create scores with char offsets that point to specific locations
    p0 = Phrase(text="Be helpful.", index=0, region_type=RegionType.PLAIN, char_start=0, char_end=11, source_text="Be helpful.")
    p1 = Phrase(text="Be concise.", index=1, region_type=RegionType.PLAIN, char_start=12, char_end=23, source_text="Be concise.")
    p2 = Phrase(text="Be helpful.", index=2, region_type=RegionType.PLAIN, char_start=24, char_end=35, source_text="Be helpful.")
    scores = [
        SaliencyScore(phrase=p0, score=0.9, raw_shapley=0.9, disposition="keep"),
        SaliencyScore(phrase=p1, score=0.05, raw_shapley=0.05, disposition="remove"),
        SaliencyScore(phrase=p2, score=0.8, raw_shapley=0.8, disposition="keep"),
    ]
    # Remove only phrase 1 (the second "Be concise.")
    diff = [
        {"phrase": 0, "action": "keep", "original": "Be helpful.", "result": "Be helpful."},
        {"phrase": 1, "action": "remove", "original": "Be concise.", "result": ""},
        {"phrase": 2, "action": "keep", "original": "Be helpful.", "result": "Be helpful."},
    ]
    result = reconstruct_from_original(original, diff, scores)
    # Both "Be helpful." should remain, "Be concise." removed
    assert result.count("Be helpful.") == 2
    assert "Be concise." not in result


def test_reconstruct_legacy_fallback():
    """scores=None triggers str.replace path (no warning since scores is None)."""
    original = "You are a chef. You are helpful."
    diff = [{"phrase": 0, "action": "remove", "original": "You are a chef.", "result": ""}]
    result = reconstruct_from_original(original, diff, scores=None)
    assert "You are a chef." not in result
    assert "You are helpful." in result


def test_validate_coverage_below_threshold():
    """Returns < 0.8 when phrases missing from diff."""
    scores = _scores("A", "B", "C", "D", "E")
    # Only 3 of 5 matched
    diff = [
        {"phrase": 0, "action": "keep", "original": "A", "result": "A"},
        {"phrase": 1, "action": "remove", "original": "B", "result": ""},
        {"phrase": 2, "action": "keep", "original": "C", "result": "C"},
    ]
    coverage = _validate_coverage(diff, scores)
    assert coverage < 0.8
    assert coverage == pytest.approx(3 / 5)
