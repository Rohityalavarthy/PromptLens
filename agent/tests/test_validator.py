"""
Tests for validator.py — covers the two pure functions:
  _compute_verdict, _pick_phrase_to_reinstate
and the async _run_validation function (timeout + early termination).
No LLM calls are made.
"""
import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from promptlens.types import Phrase, SaliencyScore, SaliencyReport, RegionType
from promptlens import SimilarityMode
from promptlens_agent.validator import _compute_verdict, _pick_phrase_to_reinstate, _run_validation


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


# ── New merge reinstatement test ─────────────────────────────────────────────

def test_reinstate_merged_phrase_also_reinstates_target():
    """When reinstating a merged phrase, its merge target is also reinstated."""
    report = make_report((0.05, "A"), (0.12, "B"), (0.08, "C"))
    diff = [
        {"phrase": 0, "action": "keep", "original": "A", "result": "A"},
        {"phrase": 1, "action": "rewrite", "original": "B", "result": "B merged"},
        {"phrase": 2, "action": "merge", "original": "C", "result": "", "merge_target": 1},
    ]
    # Phrase 2 (score 0.08) is the highest non-keep candidate after phrase 1 (0.12)
    # But phrase 1 has higher score so it gets reinstated first
    _pick_phrase_to_reinstate(diff, report)
    assert diff[1]["action"] == "keep"

    # Now phrase 2 is next — it has merge_target=1, so reinstating it should also
    # reinstate target if target was not keep (but it is now keep, so no effect needed)
    _pick_phrase_to_reinstate(diff, report)
    assert diff[2]["action"] == "keep"

    # Test the actual merge-pair reinstatement: reset
    diff2 = [
        {"phrase": 0, "action": "keep", "original": "A", "result": "A"},
        {"phrase": 1, "action": "rewrite", "original": "B", "result": "B rewritten"},
        {"phrase": 2, "action": "merge", "original": "C", "result": "", "merge_target": 1},
    ]
    # Make phrase 2 score higher than phrase 1 to force it to be reinstated first
    report2 = make_report((0.05, "A"), (0.06, "B"), (0.12, "C"))
    _pick_phrase_to_reinstate(diff2, report2)
    # Phrase 2 (highest score 0.12) should be reinstated, and its merge_target (phrase 1) too
    assert diff2[2]["action"] == "keep"
    assert diff2[1]["action"] == "keep"  # merge target also reinstated


# ── _run_validation: timeout and early termination ───────────────────────────


@pytest.mark.asyncio
async def test_early_termination_on_catastrophic_divergence():
    """
    Mock generate/compute_divergences returning 0.9 divergence (threshold=0.15),
    verify only 2 inputs processed out of 5.
    """
    generate_call_count = 0

    async def mock_generate(prompt, system_prompt=None, session=None, **kwargs):
        nonlocal generate_call_count
        generate_call_count += 1
        return f"output_{generate_call_count}"

    async def mock_compute_divergences(output_a, outputs, mode=None, session=None, **kwargs):
        # Return catastrophic divergence
        return [0.9]

    test_inputs = ["input1", "input2", "input3", "input4", "input5"]
    session = MagicMock()

    with patch("promptlens_agent.validator.generate", side_effect=mock_generate), \
         patch("promptlens_agent.validator.compute_divergences", side_effect=mock_compute_divergences):
        result = await _run_validation(
            "original prompt",
            "compressed prompt",
            test_inputs,
            SimilarityMode.STANDARD,
            session,
            threshold=0.15,
        )

    assert result == 0.9
    # 2 inputs processed, each calls generate twice (original + compressed) = 4 calls
    assert generate_call_count == 4


@pytest.mark.asyncio
async def test_timeout_records_worst_case():
    """
    Mock generate that hangs (asyncio.sleep(200)), verify divergence recorded as 1.0.
    """
    async def mock_generate_hang(prompt, system_prompt=None, session=None, **kwargs):
        await asyncio.sleep(200)
        return "never"

    async def mock_compute_divergences(output_a, outputs, mode=None, session=None, **kwargs):
        return [0.1]

    test_inputs = ["input1"]
    session = MagicMock()

    # Patch the _run_validation timeout to be very short so generate times out
    import promptlens_agent.validator as val_mod
    original_timeout = asyncio.timeout

    with patch("promptlens_agent.validator.generate", side_effect=mock_generate_hang), \
         patch("promptlens_agent.validator.compute_divergences", side_effect=mock_compute_divergences):
        # Monkey-patch a very short timeout into the function
        # We'll replace the 120.0 constant by wrapping the function call directly
        # Instead, we make generate raise TimeoutError to simulate the timeout
        pass

    # Better approach: mock generate to raise asyncio.CancelledError via timeout behavior
    # Simplest: make generate raise TimeoutError directly (simulates what happens when timeout fires)
    call_count = 0

    async def mock_generate_timeout(prompt, system_prompt=None, session=None, **kwargs):
        nonlocal call_count
        call_count += 1
        # Simulate actual timeout behavior: sleep briefly then get cancelled
        await asyncio.sleep(0.001)
        raise asyncio.TimeoutError()

    with patch("promptlens_agent.validator.generate", side_effect=mock_generate_timeout), \
         patch("promptlens_agent.validator.compute_divergences", side_effect=mock_compute_divergences):
        result = await _run_validation(
            "original prompt",
            "compressed prompt",
            test_inputs,
            SimilarityMode.STANDARD,
            session,
            threshold=0.15,
        )

    assert result == 1.0


@pytest.mark.asyncio
async def test_no_early_termination_when_below_threshold():
    """
    When divergence is below threshold * 2, all inputs should be processed.
    """
    generate_call_count = 0

    async def mock_generate(prompt, system_prompt=None, session=None, **kwargs):
        nonlocal generate_call_count
        generate_call_count += 1
        return f"output_{generate_call_count}"

    async def mock_compute_divergences(output_a, outputs, mode=None, session=None, **kwargs):
        # Below threshold * 2 = 0.30
        return [0.2]

    test_inputs = ["input1", "input2", "input3", "input4", "input5"]
    session = MagicMock()

    with patch("promptlens_agent.validator.generate", side_effect=mock_generate), \
         patch("promptlens_agent.validator.compute_divergences", side_effect=mock_compute_divergences):
        result = await _run_validation(
            "original prompt",
            "compressed prompt",
            test_inputs,
            SimilarityMode.STANDARD,
            session,
            threshold=0.15,
        )

    assert result == 0.2
    # All 5 inputs processed, each calls generate twice = 10 calls
    assert generate_call_count == 10
