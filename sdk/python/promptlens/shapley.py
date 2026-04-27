import asyncio
import random
import math
import aiohttp
from .types import Phrase, SaliencyScore, SaliencyReport, SimilarityMode
from .generator import generate
from .similarity import compute_divergences
from .segmenter import segment_prompt


# ── Prompt reconstruction ─────────────────────────────────────────────────────

def build_prompt(phrases: list[Phrase], active_indices: set[int]) -> str:
    """
    Reconstruct a prompt from a subset of active phrase indices.
    Preserves original phrase order. Omits inactive phrases.
    """
    parts = []
    for phrase in phrases:
        if phrase.index in active_indices:
            parts.append(phrase.text)
    return ' '.join(parts)


def build_empty_prompt(phrases: list[Phrase]) -> str:
    """Prompt with all phrases redacted — the empty coalition baseline."""
    return ' '.join('[...]' for _ in phrases)


# ── Coalition cache ───────────────────────────────────────────────────────────

class CoalitionCache:
    def __init__(self):
        self._cache: dict[str, str] = {}

    def key(self, active_indices: set[int]) -> str:
        return ','.join(str(i) for i in sorted(active_indices))

    def get(self, active_indices: set[int]) -> str | None:
        return self._cache.get(self.key(active_indices))

    def set(self, active_indices: set[int], output: str) -> None:
        self._cache[self.key(active_indices)] = output


# ── Single coalition walk ─────────────────────────────────────────────────────

async def sample_coalition_walk(
    phrases: list[Phrase],
    user_input: str,
    empty_output: str,
    cache: CoalitionCache,
    mode: SimilarityMode,
    session: aiohttp.ClientSession,
) -> list[float]:
    """
    One Monte Carlo walk: random permutation of phrases, record marginal divergence
    when each phrase is added. Returns list of marginal contributions indexed by phrase.index.
    """
    n = len(phrases)
    order = list(range(n))
    random.shuffle(order)

    contributions = [0.0] * n
    active: set[int] = set()
    prev_output = empty_output

    for phrase_pos in order:
        phrase = phrases[phrase_pos]
        active.add(phrase.index)

        # Check cache first
        cached = cache.get(active)
        if cached is not None:
            current_output = cached
        else:
            prompt = build_prompt(phrases, active)
            # System prompt goes in the system role; user input in the user role
            current_output = await generate(
                user_input if user_input else "[no input]",
                system_prompt=prompt,
                session=session,
            )
            cache.set(active.copy(), current_output)

        # Marginal contribution = how much output changed by adding this phrase
        divergences = await compute_divergences(
            prev_output, [current_output], mode=mode, session=session
        )
        contributions[phrase_pos] = divergences[0]
        prev_output = current_output

    return contributions


# ── Main Shapley analysis ─────────────────────────────────────────────────────

async def run_shapley(
    prompt: str,
    test_inputs: list[str],
    m_samples: int = 20,
    mode: SimilarityMode = SimilarityMode.STANDARD,
    low_saliency_threshold: float = 0.15,
    session: aiohttp.ClientSession | None = None,
) -> SaliencyReport:
    """
    Full Shapley attribution pipeline.

    Args:
        prompt: The system prompt to analyse.
        test_inputs: List of user inputs to test against. Shapley scores are
                     averaged across all inputs for stable multi-input estimates.
        m_samples: Number of Monte Carlo coalition walks per test input.
                   M=20 for full analysis (promptlens audit).
                   M=3 for fast mode (promptlens check).
        mode: STANDARD (trigram) or SEMANTIC (embedding + judge).
        low_saliency_threshold: Phrases below this normalised score are candidates.
        session: aiohttp session. Created internally if None.

    Returns:
        SaliencyReport with per-phrase scores and compression brief.
    """
    phrases = segment_prompt(prompt)
    n = len(phrases)

    # For very short prompts (N <= 4), compute exact Shapley (2^N <= 16 calls)
    if n <= 4:
        m_samples = 2 ** n  # exact — all permutations covered

    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        # Pre-compute empty coalition output (cache it — always the same)
        empty_prompt_text = build_empty_prompt(phrases)
        empty_output = await generate(
            test_inputs[0] if test_inputs else "[no input]",
            system_prompt=empty_prompt_text,
            session=session,
        )

        cache = CoalitionCache()
        all_contributions: list[list[float]] = []

        # Run M walks per test input, parallelise across walks (cap: 5 concurrent)
        semaphore = asyncio.Semaphore(5)

        async def run_walk(user_input: str) -> list[float]:
            async with semaphore:
                return await sample_coalition_walk(
                    phrases, user_input, empty_output, cache, mode, session
                )

        tasks = [
            run_walk(inp)
            for inp in test_inputs
            for _ in range(m_samples)
        ]
        all_contributions = await asyncio.gather(*tasks)

        # Average marginal contributions per phrase across all walks and inputs
        avg_contributions = [0.0] * n
        for walk in all_contributions:
            for i, contrib in enumerate(walk):
                avg_contributions[i] += contrib
        total_walks = len(all_contributions)
        avg_contributions = [c / total_walks for c in avg_contributions]

        # Normalise to 0.0–1.0 using min-max
        min_c = min(avg_contributions)
        max_c = max(avg_contributions)
        if max_c - min_c < 1e-9:
            normalised = [0.5] * n
        else:
            normalised = [(c - min_c) / (max_c - min_c) for c in avg_contributions]

        # Build SaliencyScore list
        scores = []
        for phrase, raw, norm in zip(phrases, avg_contributions, normalised):
            disposition = "remove" if norm < low_saliency_threshold else "keep"
            scores.append(SaliencyScore(
                phrase=phrase,
                score=norm,
                raw_shapley=raw,
                disposition=disposition,
            ))

        # Compute report statistics
        low_saliency_phrases = [s for s in scores if s.score < low_saliency_threshold]
        candidate_tokens = sum(
            len(s.phrase.text.split()) for s in low_saliency_phrases
        )
        total_tokens = sum(len(p.text.split()) for p in phrases)
        redundancy_fraction = len(low_saliency_phrases) / n if n > 0 else 0.0

        # Confidence: stddev of normalised scores (lower variance = more stable ranking)
        score_variance = sum((s - 0.5)**2 for s in normalised) / n
        confidence = max(0.0, 1.0 - math.sqrt(score_variance))

        return SaliencyReport(
            prompt=prompt,
            phrases=phrases,
            scores=scores,
            token_count=total_tokens,
            redundancy_fraction=redundancy_fraction,
            compression_candidate_tokens=candidate_tokens,
            m_samples=m_samples,
            test_inputs_used=len(test_inputs),
            confidence=round(confidence, 2),
        )

    finally:
        if close_session:
            await session.close()
