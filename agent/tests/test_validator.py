"""
Tests for validator.py — covers the two pure functions:
  _compute_verdict, _pick_phrase_to_reinstate
No LLM calls are made.
"""
import pytest
from promptlens.types import Phrase, SaliencyScore, SaliencyReport, RegionType
from promptlens_agent.validator import _compute_verdict, _pick_phrase_to_reinstate


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_report(*scores_and_texts) -> SaliencyReport:
    """Build a minimal SaliencyReport from (score, text) pairs."""
    phrases, scores = [], []
    for i, (score_val, text) in enumerate(scores_and_texts):
        p = Phrase(text=text, index=i, region_type=RegionType.PLAIN)
        s = SaliencyScore(phrase=p, score=score_val, raw_shapley=score_val, disposition="remove")
        phrases.append(p)
        scores.append(s)
    return SaliencyReport(
        prompt=" ".join(t for _, t in scores_and_texts),
        phrases=phrases,
        scores=scores,
        token_count=len(scores_and_texts),
        redundancy_fraction=1.0,
        compression_candidate_tokens=len(scores_and_texts),
        m_samples=5,
        test_inputs_used=3,
        confidence=0.8,
    )


def make_diff(*entries) -> list[dict]:
    """Build a diff list from (phrase_idx, action, original, result) tuples."""
    return [
        {"phrase": idx, "action": action, "original": orig, "result": res}
        for idx, action, orig, res in entries
    ]


# ── _compute_verdict ──────────────────────────────────────────────────────────

def test_verdict_pass_below_threshold():
    assert _compute_verdict(0.10, 0.20) == "PASS"

def test_verdict_pass_exactly_at_threshold():
    assert _compute_verdict(0.20, 0.20) == "PASS"

def test_verdict_marginal_just_over():
    # 1.1× threshold — within the 1.25× MARGINAL band
    assert _compute_verdict(0.22, 0.20) == "MARGINAL"

def test_verdict_marginal_at_upper_bound():
    # exactly 1.25× threshold
    assert _compute_verdict(0.25, 0.20) == "MARGINAL"

def test_verdict_review_above_marginal():
    # 1.5× threshold — within the 1.75× REVIEW band
    assert _compute_verdict(0.30, 0.20) == "REVIEW"

def test_verdict_review_at_upper_bound():
    # exactly 1.75× threshold
    assert _compute_verdict(0.35, 0.20) == "REVIEW"

def test_verdict_fail_above_review():
    # 2.0× threshold — beyond 1.75×
    assert _compute_verdict(0.40, 0.20) == "FAIL"

def test_verdict_scales_with_threshold():
    # Same ratio, different threshold — verdict should match
    assert _compute_verdict(0.40, 0.40) == "PASS"
    assert _compute_verdict(0.50, 0.40) == "MARGINAL"
    assert _compute_verdict(0.60, 0.40) == "REVIEW"
    assert _compute_verdict(0.80, 0.40) == "FAIL"


# ── _pick_phrase_to_reinstate ─────────────────────────────────────────────────

def test_reinstate_picks_highest_score():
    # phrase 0 scores 0.05, phrase 1 scores 0.12 — should reinstate phrase 1 first
    report = make_report((0.05, "You are clear."), (0.12, "You are professional."))
    diff = make_diff(
        (0, "remove", "You are clear.",        ""),
        (1, "remove", "You are professional.", ""),
    )
    _pick_phrase_to_reinstate(diff, report)
    assert diff[1]["action"] == "keep"     # higher score reinstated
    assert diff[0]["action"] == "remove"   # lower score left alone


def test_reinstate_sets_result_to_original():
    report = make_report((0.08, "You are helpful."))
    diff = make_diff((0, "paraphrase", "You are helpful.", "Be helpful."))
    _pick_phrase_to_reinstate(diff, report)
    assert diff[0]["result"] == "You are helpful."


def test_reinstate_works_for_all_non_keep_actions():
    for action in ("remove", "rewrite", "merge", "paraphrase"):
        report = make_report((0.08, "phrase"))
        diff = make_diff((0, action, "phrase", "result"))
        result = _pick_phrase_to_reinstate(diff, report)
        assert result is True
        assert diff[0]["action"] == "keep"


def test_reinstate_skips_already_kept_phrases():
    report = make_report((0.12, "You are clear."), (0.08, "You are helpful."))
    diff = make_diff(
        (0, "keep",   "You are clear.",   "You are clear."),
        (1, "remove", "You are helpful.", ""),
    )
    _pick_phrase_to_reinstate(diff, report)
    assert diff[1]["action"] == "keep"
    assert diff[0]["action"] == "keep"  # was already kept — unchanged


def test_reinstate_returns_false_when_nothing_to_reinstate():
    report = make_report((0.10, "You are a chef."))
    diff = make_diff((0, "keep", "You are a chef.", "You are a chef."))
    assert _pick_phrase_to_reinstate(diff, report) is False


def test_reinstate_returns_false_on_empty_diff():
    report = make_report((0.10, "phrase"))
    assert _pick_phrase_to_reinstate([], report) is False


def test_reinstate_handles_missing_phrase_index_gracefully():
    # phrase index out of range — falls back to score 0.0, still reinstates
    report = make_report((0.10, "phrase"))
    diff = [{"phrase": 99, "action": "remove", "original": "ghost phrase", "result": ""}]
    result = _pick_phrase_to_reinstate(diff, report)
    assert result is True
    assert diff[0]["action"] == "keep"


def test_reinstate_successive_calls_work_through_candidates():
    # Three phrases removed; each call should reinstate the next highest-score one
    report = make_report((0.03, "A"), (0.11, "B"), (0.07, "C"))
    diff = make_diff(
        (0, "remove", "A", ""),
        (1, "remove", "B", ""),
        (2, "remove", "C", ""),
    )
    _pick_phrase_to_reinstate(diff, report)
    assert diff[1]["action"] == "keep"    # score 0.11 — first

    _pick_phrase_to_reinstate(diff, report)
    assert diff[2]["action"] == "keep"    # score 0.07 — second

    _pick_phrase_to_reinstate(diff, report)
    assert diff[0]["action"] == "keep"    # score 0.03 — last
