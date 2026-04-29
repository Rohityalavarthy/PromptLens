import asyncio
import aiohttp
from promptlens import SaliencyReport, SimilarityMode
from promptlens.generator import generate
from promptlens.similarity import compute_divergences
from .compressor import reconstruct_from_original

MAX_RETRIES = 3


async def validate_compression(
    original_prompt: str,
    compressed_prompt: str,
    test_inputs: list[str],
    report: SaliencyReport,
    diff: list[dict],
    threshold: float = 0.15,
    mode: SimilarityMode = SimilarityMode.STANDARD,
) -> tuple[bool, float, str]:
    """
    Run compressed prompt against all test inputs.
    Compare outputs to original using divergence threshold.

    Returns:
        (passed, worst_case_divergence, final_compressed_prompt)

    If validation fails, iteratively reinstate high-divergence phrases and retry
    up to MAX_RETRIES times.
    """
    async with aiohttp.ClientSession() as session:
        current_compressed = compressed_prompt
        current_diff = diff

        for attempt in range(MAX_RETRIES):
            worst_divergence, failing_phrase_indices = await _run_validation(
                original_prompt, current_compressed, test_inputs, threshold, mode, session
            )

            if worst_divergence <= threshold:
                return True, worst_divergence, current_compressed

            if not failing_phrase_indices:
                break

            # Reinstate the phrase with highest divergence signal
            reinstated = False
            for phrase_idx in failing_phrase_indices:
                if phrase_idx < len(current_diff):
                    entry = current_diff[phrase_idx]
                    if entry["action"] in ("remove", "rewrite", "merge"):
                        entry["action"] = "keep"
                        entry["result"] = entry["original"]
                        reinstated = True
                        break

            if not reinstated:
                break

            # Rebuild compressed prompt from updated diff, preserving formatting
            current_compressed = reconstruct_from_original(original_prompt, current_diff)

        # Final check
        worst_divergence, _ = await _run_validation(
            original_prompt, current_compressed, test_inputs, threshold, mode, session
        )
        passed = worst_divergence <= threshold
        return passed, worst_divergence, current_compressed


async def _run_validation(
    original_prompt: str,
    compressed_prompt: str,
    test_inputs: list[str],
    threshold: float,
    mode: SimilarityMode,
    session: aiohttp.ClientSession,
) -> tuple[float, list[int]]:
    """
    Run both prompts against all test inputs, return worst divergence
    and indices of test inputs that caused divergence above threshold.
    """
    divergences = []
    for user_input in test_inputs:
        original_output = await generate(user_input, system_prompt=original_prompt, session=session)
        compressed_output = await generate(user_input, system_prompt=compressed_prompt, session=session)
        divs = await compute_divergences(original_output, [compressed_output], mode=mode, session=session)
        divergences.append(divs[0])

    worst = max(divergences)
    failing = [i for i, d in enumerate(divergences) if d > threshold]
    return worst, failing
