import asyncio
import logging

import aiohttp
from collections.abc import Callable
from promptlens import SaliencyReport, SimilarityMode
from promptlens.generator import generate
from promptlens.similarity import compute_divergences
from .compressor import reconstruct_from_original

logger = logging.getLogger("promptlens.validator")

MAX_RETRIES = 3

# Verdict thresholds expressed as multiples of the user's chosen threshold.
# Anything at or below 1× is a clean pass; above 1.75× is a hard fail.
_MARGINAL_RATIO = 1.25
_REVIEW_RATIO   = 1.75


def _compute_verdict(worst_divergence: float, threshold: float) -> str:
    """
    Map a worst-case divergence value to a human verdict.

    PASS     — within the threshold gate, safe to adopt
    MARGINAL — slightly over; spot-check a few outputs
    REVIEW   — noticeably over; review changes carefully before adopting
    FAIL     — significantly over; do not apply without thorough review
    """
    if worst_divergence <= threshold:
        return "PASS"
    ratio = worst_divergence / threshold if threshold > 0 else float("inf")
    if ratio <= _MARGINAL_RATIO:
        return "MARGINAL"
    if ratio <= _REVIEW_RATIO:
        return "REVIEW"
    return "FAIL"


def _pick_phrase_to_reinstate(diff: list[dict], report: SaliencyReport) -> bool:
    """
    Find the highest-Shapley-score modified phrase in diff and mark it as kept.

    Sorting by score descending means we reinstate the phrase most likely to
    have caused output divergence first — it had the most impact when present,
    so removing or rewriting it is the riskiest change.

    For merge pairs: when reinstating a merged phrase, also reinstate its merge target.

    Returns True if a phrase was reinstated, False if every phrase is already kept.
    """
    candidates = []
    for entry in diff:
        if entry["action"] == "keep":
            continue
        phrase_idx = entry.get("phrase", -1)
        score = (
            report.scores[phrase_idx].score
            if 0 <= phrase_idx < len(report.scores)
            else 0.0
        )
        candidates.append((score, entry))

    if not candidates:
        return False

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, entry = candidates[0]
    entry["action"] = "keep"
    entry["result"] = entry["original"]

    # If this was a merge entry, also reinstate its merge target
    merge_target = entry.get("merge_target")
    if merge_target is not None:
        for other_entry in diff:
            if other_entry.get("phrase") == merge_target and other_entry["action"] != "keep":
                other_entry["action"] = "keep"
                other_entry["result"] = other_entry["original"]
                break

    return True


async def validate_compression(
    original_prompt: str,
    compressed_prompt: str,
    test_inputs: list[str],
    report: SaliencyReport,
    diff: list[dict],
    threshold: float = 0.15,
    mode: SimilarityMode = SimilarityMode.STANDARD,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[str, float, str]:
    """
    Run compressed prompt against all test inputs and compare outputs to original.

    Returns:
        (verdict, worst_case_divergence, final_compressed_prompt)

    verdict is one of: "PASS" | "MARGINAL" | "REVIEW" | "FAIL"

    On failure, iteratively reinstates the highest-score modified phrase and
    retries up to MAX_RETRIES times before returning the best result found.
    """
    async with aiohttp.ClientSession() as session:
        current_compressed = compressed_prompt
        current_diff = diff

        for _ in range(MAX_RETRIES):
            worst_divergence = await _run_validation(
                original_prompt, current_compressed, test_inputs, mode, session, on_progress,
                threshold=threshold,
            )

            if worst_divergence <= threshold:
                return "PASS", worst_divergence, current_compressed

            reinstated = _pick_phrase_to_reinstate(current_diff, report)
            if not reinstated:
                break

            current_compressed = reconstruct_from_original(original_prompt, current_diff, report.scores)

        # Final measurement after all reinstatements
        worst_divergence = await _run_validation(
            original_prompt, current_compressed, test_inputs, mode, session, on_progress,
            threshold=threshold,
        )
        verdict = _compute_verdict(worst_divergence, threshold)
        return verdict, worst_divergence, current_compressed


async def _run_validation(
    original_prompt: str,
    compressed_prompt: str,
    test_inputs: list[str],
    mode: SimilarityMode,
    session: aiohttp.ClientSession,
    on_progress: Callable[[int, int], None] | None = None,
    threshold: float = 0.15,
) -> float:
    """
    Run both prompts against all test inputs and return the worst divergence seen.

    Features:
    - Per-input timeout (120s): records worst-case divergence (1.0) on timeout
    - Early termination: if first 2 inputs both show catastrophic divergence
      (> threshold * 2), stops early and returns worst score
    """
    worst = 0.0
    early_failures = 0

    for i, user_input in enumerate(test_inputs):
        try:
            async with asyncio.timeout(120.0):
                original_output = await generate(user_input, system_prompt=original_prompt, session=session)
                compressed_output = await generate(user_input, system_prompt=compressed_prompt, session=session)
                divs = await compute_divergences(original_output, [compressed_output], mode=mode, session=session)
                divergence = divs[0]
        except asyncio.TimeoutError:
            logger.warning(
                f"Validation timeout on input {i + 1}/{len(test_inputs)}, recording worst-case divergence"
            )
            divergence = 1.0

        worst = max(worst, divergence)

        if on_progress:
            on_progress(i + 1, len(test_inputs))

        # Early termination on catastrophic divergence
        if i < 2 and divergence > threshold * 2:
            early_failures += 1
        if early_failures >= 2 and i == 1:
            logger.info(
                f"Early termination: first 2 inputs show catastrophic divergence ({worst:.3f})"
            )
            return worst

    return worst
