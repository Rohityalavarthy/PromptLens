import math
import asyncio
import aiohttp
from .generator import get_embedding, judge_divergence
from .types import SimilarityMode


# ── Standard: trigram cosine ──────────────────────────────────────────────────

def get_trigrams(text: str) -> dict[str, int]:
    trigrams: dict[str, int] = {}
    for i in range(len(text) - 2):
        t = text[i:i+3]
        trigrams[t] = trigrams.get(t, 0) + 1
    return trigrams


def trigram_similarity(a: str, b: str) -> float:
    """Returns 0.0 (identical) to 1.0 (completely different)."""
    tg_a = get_trigrams(a)
    tg_b = get_trigrams(b)
    all_keys = set(tg_a) | set(tg_b)
    if not all_keys:
        return 0.0
    dot = sum(tg_a.get(k, 0) * tg_b.get(k, 0) for k in all_keys)
    mag_a = math.sqrt(sum(v**2 for v in tg_a.values()))
    mag_b = math.sqrt(sum(v**2 for v in tg_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    cosine_similarity = dot / (mag_a * mag_b)
    return 1.0 - cosine_similarity   # distance = divergence


# ── Semantic: embedding cosine + z-score judge ────────────────────────────────

def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a**2 for a in vec_a))
    mag_b = math.sqrt(sum(b**2 for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1.0 - (dot / (mag_a * mag_b))


async def semantic_divergence(
    baseline: str,
    perturbed_outputs: list[str],
    baseline_embedding: list[float],
    session: aiohttp.ClientSession,
) -> list[float]:
    """
    Two-pass hybrid:
    Pass 1 — embedding cosine distance for all phrases.
    Pass 2 — z-score threshold: phrases > mean+stddev go to judge.
    Returns list of final divergence scores (0.0–1.0) per perturbed output.
    """
    # Pass 1: embed all perturbed outputs
    perturbed_embeddings = await asyncio.gather(*[
        get_embedding(output, session) for output in perturbed_outputs
    ])
    embedding_distances = [
        cosine_distance(baseline_embedding, pe)
        for pe in perturbed_embeddings
    ]

    # Z-score threshold
    mean = sum(embedding_distances) / len(embedding_distances)
    variance = sum((d - mean)**2 for d in embedding_distances) / len(embedding_distances)
    std_dev = math.sqrt(variance)
    threshold = mean + std_dev

    # Guard: if stddev near zero, judge the single highest-scoring phrase
    judge_indices: set[int]
    if std_dev < 0.01:
        judge_indices = {embedding_distances.index(max(embedding_distances))}
    else:
        judge_indices = {i for i, d in enumerate(embedding_distances) if d > threshold}

    # Pass 2: judge on outliers (concurrency cap = 3)
    semaphore = asyncio.Semaphore(3)
    final_scores = list(embedding_distances)  # copy

    async def judge_one(i: int) -> None:
        async with semaphore:
            judge_score = await judge_divergence(baseline, perturbed_outputs[i], session)
            normalized_judge = judge_score / 10.0
            final_scores[i] = (embedding_distances[i] + normalized_judge) / 2

    await asyncio.gather(*[judge_one(i) for i in judge_indices])
    return final_scores


# ── Unified interface ─────────────────────────────────────────────────────────

async def compute_divergences(
    baseline: str,
    perturbed_outputs: list[str],
    mode: SimilarityMode = SimilarityMode.STANDARD,
    session: aiohttp.ClientSession | None = None,
) -> list[float]:
    """
    Returns divergence score per perturbed output (0.0 = identical, 1.0 = completely different).
    In STANDARD mode: no session needed.
    In SEMANTIC mode: session required (Together AI embedding + judge calls).
    """
    if mode == SimilarityMode.STANDARD:
        return [trigram_similarity(baseline, p) for p in perturbed_outputs]

    # SEMANTIC mode
    assert session is not None, "aiohttp session required for SEMANTIC mode"
    baseline_embedding = await get_embedding(baseline, session)
    return await semantic_divergence(baseline, perturbed_outputs, baseline_embedding, session)
