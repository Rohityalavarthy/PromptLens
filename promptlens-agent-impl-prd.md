# PromptLens Agent — Implementation PRD
**For:** Claude Code  
**Scope:** Python CLI tool + Python SDK (Option A — Python port of core engine, Entry Point 1 — CLI)  
**Repo:** Extend existing `Rohityalavarthy/PromptLens` as a monorepo  
**Stack:** Python 3.11+, Together AI API, LangGraph, Click  
**Do not touch:** `web/` directory (existing browser tool stays unchanged)

---

## Monorepo Structure to Create

```
PromptLens/
├── web/                          ← EXISTING — do not modify
│   ├── index.html
│   └── app.js
├── sdk/
│   └── python/
│       ├── promptlens/
│       │   ├── __init__.py
│       │   ├── segmenter.py      ← phrase segmentation (port from app.js)
│       │   ├── shapley.py        ← shapley engine (port from app.js)
│       │   ├── similarity.py     ← similarity scoring (port from app.js)
│       │   ├── generator.py      ← LLM API calls (Together AI)
│       │   └── types.py          ← dataclasses / typed dicts
│       ├── tests/
│       │   ├── test_segmenter.py
│       │   ├── test_shapley.py
│       │   └── test_similarity.py
│       └── pyproject.toml
├── agent/
│   ├── promptlens_agent/
│   │   ├── __init__.py
│   │   ├── cli.py                ← Click CLI entry point
│   │   ├── discovery.py          ← codebase prompt discovery
│   │   ├── compressor.py         ← constrained LLM rewrite
│   │   ├── validator.py          ← compression validation loop
│   │   ├── reporter.py           ← terminal output formatting
│   │   └── tools.py              ← LangGraph tool definitions
│   ├── tests/
│   │   └── test_discovery.py
│   └── pyproject.toml
└── README.md
```

---

## Part 1 — Python SDK (`sdk/python/`)

### 1.1 `types.py` — All shared types

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class RegionType(Enum):
    PLAIN = "plain"
    CODE_BLOCK = "code_block"
    JSON = "json"
    BULLETS = "bullets"
    XML_TAGGED = "xml_tagged"
    MARKDOWN = "markdown"

class SimilarityMode(Enum):
    STANDARD = "standard"   # trigram cosine
    SEMANTIC = "semantic"   # embedding + judge

@dataclass
class Region:
    type: RegionType
    text: str

@dataclass
class Phrase:
    text: str
    index: int
    atomic: bool = False
    region_type: RegionType = RegionType.PLAIN
    tag_name: Optional[str] = None       # for XML phrases — used during reconstruction

@dataclass
class SaliencyScore:
    phrase: Phrase
    score: float                          # 0.0 to 1.0, normalised
    raw_shapley: float                    # unnormalised average marginal contribution
    disposition: str = "keep"            # keep | remove | merge | rewrite

@dataclass
class SaliencyReport:
    prompt: str
    phrases: list[Phrase]
    scores: list[SaliencyScore]
    token_count: int
    redundancy_fraction: float           # fraction of phrases scoring below threshold
    compression_candidate_tokens: int    # tokens in low-saliency phrases
    m_samples: int
    test_inputs_used: int
    confidence: float                    # average score stability across inputs

@dataclass
class CompressionResult:
    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    token_delta: int
    validation_passed: bool
    worst_case_divergence: float
    saliency_report: SaliencyReport
    diff: list[dict]                     # list of {phrase, action, original, compressed}
```

---

### 1.2 `segmenter.py` — Structure-aware phrase segmentation

Port of the segmentation logic designed in the segmentation PRD. Detects structural regions in a prompt and routes each to the correct segmenter.

```python
import re
import json
from .types import Phrase, Region, RegionType

# ── Region Detection ──────────────────────────────────────────────────────────

def detect_regions(prompt: str) -> list[Region]:
    """
    Scan prompt top-to-bottom, classify contiguous text regions by structural type.
    Returns ordered list of regions. Every character belongs to exactly one region.
    """
    regions = []
    lines = prompt.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Code fence block — ```...```
        if line.strip().startswith('```'):
            block_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                block_lines.append(lines[i])
                i += 1
            if i < len(lines):
                block_lines.append(lines[i])
            regions.append(Region(type=RegionType.CODE_BLOCK, text='\n'.join(block_lines)))
            i += 1
            continue

        # JSON block — line starts with { or [
        if line.strip().startswith(('{', '[')):
            block_lines = [line]
            i += 1
            # collect until matching close brace (simple depth counter)
            depth = line.count('{') + line.count('[') - line.count('}') - line.count(']')
            while i < len(lines) and depth > 0:
                block_lines.append(lines[i])
                depth += lines[i].count('{') + lines[i].count('[')
                depth -= lines[i].count('}') + lines[i].count(']')
                i += 1
            regions.append(Region(type=RegionType.JSON, text='\n'.join(block_lines)))
            continue

        # Bullet list — lines starting with -, *, •, or N.
        if re.match(r'^\s*([-*•]|\d+\.)\s', line):
            block_lines = []
            while i < len(lines) and (re.match(r'^\s*([-*•]|\d+\.)\s', lines[i]) or
                                       (block_lines and lines[i].startswith('  '))):
                block_lines.append(lines[i])
                i += 1
            regions.append(Region(type=RegionType.BULLETS, text='\n'.join(block_lines)))
            continue

        # Markdown section — ## or ### header
        if re.match(r'^#{1,3}\s', line):
            block_lines = [line]
            i += 1
            while i < len(lines) and not re.match(r'^#{1,3}\s', lines[i]):
                block_lines.append(lines[i])
                i += 1
            regions.append(Region(type=RegionType.MARKDOWN, text='\n'.join(block_lines)))
            continue

        # XML tagged — line contains <tag>
        if re.search(r'<[a-zA-Z_][a-zA-Z0-9_]*>', line):
            block_lines = [line]
            i += 1
            while i < len(lines) and re.search(r'</?[a-zA-Z_][a-zA-Z0-9_]*>', lines[i]):
                block_lines.append(lines[i])
                i += 1
            regions.append(Region(type=RegionType.XML_TAGGED, text='\n'.join(block_lines)))
            continue

        # Plain text — collect until next structural boundary
        block_lines = [line]
        i += 1
        while i < len(lines) and not _is_structural_start(lines[i]):
            block_lines.append(lines[i])
            i += 1
        text = '\n'.join(block_lines).strip()
        if text:
            regions.append(Region(type=RegionType.PLAIN, text=text))

    return [r for r in regions if r.text.strip()]


def _is_structural_start(line: str) -> bool:
    return bool(
        line.strip().startswith('```') or
        line.strip().startswith(('{', '[')) or
        re.match(r'^\s*([-*•]|\d+\.)\s', line) or
        re.match(r'^#{1,3}\s', line) or
        re.search(r'<[a-zA-Z_][a-zA-Z0-9_]*>', line)
    )


# ── Per-type segmenters ────────────────────────────────────────────────────────

def segment_plain_text(text: str, start_index: int = 0) -> list[Phrase]:
    """
    Split on sentence boundaries (. ! ?) then clause boundaries (, ; :).
    Apply 35-60 char heuristic. Non-atomic.
    """
    MIN_CHARS, MAX_CHARS = 35, 60
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    phrases = []
    idx = start_index

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= MAX_CHARS:
            phrases.append(Phrase(text=sentence, index=idx, atomic=False, region_type=RegionType.PLAIN))
            idx += 1
        else:
            clauses = re.split(r'(?<=[,;:])\s+', sentence)
            current = ''
            for clause in clauses:
                if len(current) + len(clause) < MAX_CHARS:
                    current = (current + ' ' + clause).strip()
                else:
                    if len(current) >= MIN_CHARS:
                        phrases.append(Phrase(text=current, index=idx, atomic=False, region_type=RegionType.PLAIN))
                        idx += 1
                        current = clause
                    else:
                        current = (current + ' ' + clause).strip()
            if current:
                phrases.append(Phrase(text=current, index=idx, atomic=False, region_type=RegionType.PLAIN))
                idx += 1

    return phrases


def segment_code_block(text: str, start_index: int = 0) -> list[Phrase]:
    """Entire code block is one atomic phrase."""
    return [Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.CODE_BLOCK)]


def segment_json(text: str, start_index: int = 0) -> list[Phrase]:
    """
    Parse JSON and treat each top-level key-value pair as one phrase.
    Fall back to atomic on parse failure.
    """
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return [Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.JSON)]
        phrases = []
        for i, (key, value) in enumerate(parsed.items()):
            phrase_text = f'"{key}": {json.dumps(value)}'
            phrases.append(Phrase(text=phrase_text, index=start_index + i, atomic=False, region_type=RegionType.JSON))
        return phrases
    except (json.JSONDecodeError, ValueError):
        return [Phrase(text=text, index=start_index, atomic=True, region_type=RegionType.JSON)]


def segment_bullets(text: str, start_index: int = 0) -> list[Phrase]:
    """Each bullet item (including its continuation lines) is one phrase."""
    phrases = []
    idx = start_index
    current_bullet = None

    for line in text.split('\n'):
        if re.match(r'^\s*([-*•]|\d+\.)\s', line):
            if current_bullet:
                phrases.append(Phrase(text=current_bullet.strip(), index=idx, atomic=False, region_type=RegionType.BULLETS))
                idx += 1
            current_bullet = line
        elif current_bullet and line.startswith('  '):
            current_bullet += ' ' + line.strip()

    if current_bullet:
        phrases.append(Phrase(text=current_bullet.strip(), index=idx, atomic=False, region_type=RegionType.BULLETS))

    return phrases


def segment_xml(text: str, start_index: int = 0) -> list[Phrase]:
    """Each <tag>content</tag> pair is one phrase. Preserves tag_name for reconstruction."""
    pattern = re.compile(r'<(\w+)>([\s\S]*?)</\1>', re.DOTALL)
    phrases = []
    idx = start_index

    for match in pattern.finditer(text):
        phrases.append(Phrase(
            text=match.group(0),
            index=idx,
            atomic=False,
            region_type=RegionType.XML_TAGGED,
            tag_name=match.group(1),
        ))
        idx += 1

    if not phrases:
        return segment_plain_text(text, start_index)
    return phrases


def segment_markdown(text: str, start_index: int = 0) -> list[Phrase]:
    """
    Split at ## / ### headers. Header is atomic. Body uses plain text segmenter.
    """
    sections = re.split(r'(?=^#{1,3}\s)', text, flags=re.MULTILINE)
    phrases = []
    idx = start_index

    for section in sections:
        if not section.strip():
            continue
        header_match = re.match(r'^(#{1,3}\s.+)\n?', section)
        if header_match:
            phrases.append(Phrase(
                text=header_match.group(1).strip(),
                index=idx,
                atomic=True,
                region_type=RegionType.MARKDOWN,
            ))
            idx += 1
            body = section[len(header_match.group(0)):]
            if body.strip():
                body_phrases = segment_plain_text(body.strip(), idx)
                phrases.extend(body_phrases)
                idx += len(body_phrases)
        else:
            plain = segment_plain_text(section.strip(), idx)
            phrases.extend(plain)
            idx += len(plain)

    return phrases


# ── Main entry point ──────────────────────────────────────────────────────────

def segment_prompt(prompt: str) -> list[Phrase]:
    """
    Full pipeline: detect regions → route to segmenter → return flat phrase list.
    Atomic phrases are never split by character count.
    """
    regions = detect_regions(prompt)
    phrases = []
    idx = 0

    segmenters = {
        RegionType.PLAIN:      segment_plain_text,
        RegionType.CODE_BLOCK: segment_code_block,
        RegionType.JSON:       segment_json,
        RegionType.BULLETS:    segment_bullets,
        RegionType.XML_TAGGED: segment_xml,
        RegionType.MARKDOWN:   segment_markdown,
    }

    for region in regions:
        fn = segmenters.get(region.type, segment_plain_text)
        region_phrases = fn(region.text, start_index=idx)
        phrases.extend(region_phrases)
        idx += len(region_phrases)

    return phrases
```

---

### 1.3 `generator.py` — LLM + Embedding API calls

All API calls go to Together AI. Single API key. Generator uses Llama, embedder uses `nomic-embed-text`, judge uses Qwen (different architecture from generator — cross-model judge independence).

```python
import os
import asyncio
import aiohttp
from typing import Optional
from .types import SimilarityMode

TOGETHER_API_BASE = "https://api.together.xyz/v1"

GENERATOR_MODEL  = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
EMBEDDING_MODEL  = "togethercomputer/m2-bert-80M-8k-retrieval"  # fallback: nomic-embed-text-v1.5
JUDGE_MODEL      = "Qwen/Qwen2.5-72B-Instruct-Turbo"


def get_api_key() -> str:
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise EnvironmentError(
            "TOGETHER_API_KEY environment variable not set.\n"
            "Get a key at https://api.together.xyz and run:\n"
            "  export TOGETHER_API_KEY=your_key_here"
        )
    return key


async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    session: Optional[aiohttp.ClientSession] = None,
) -> str:
    """Single LLM generation call. Returns response text."""
    api_key = get_api_key()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GENERATOR_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    close_session = session is None
    if session is None:
        session = aiohttp.ClientSession()

    try:
        async with session.post(
            f"{TOGETHER_API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    finally:
        if close_session:
            await session.close()


async def get_embedding(text: str, session: aiohttp.ClientSession) -> list[float]:
    """Get text embedding from Together AI. Returns float vector."""
    api_key = get_api_key()
    async with session.post(
        f"{TOGETHER_API_BASE}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": EMBEDDING_MODEL, "input": text},
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data["data"][0]["embedding"]


async def judge_divergence(
    output_a: str,
    output_b: str,
    session: aiohttp.ClientSession,
) -> int:
    """
    Ask Qwen to rate semantic divergence between two outputs.
    Returns integer 0-10. 0 = identical meaning, 10 = completely different.
    Uses different model family from generator (Qwen vs Llama) for independence.
    """
    api_key = get_api_key()
    prompt = f"""You are a semantic equivalence evaluator.

Rate how semantically different these two responses are — not surface differences, but differences in meaning, recommendations, facts, or intent.

Response A:
{output_a}

Response B:
{output_b}

Return a single integer from 0 to 10. 0 = identical meaning. 10 = completely different. Return only the number."""

    async with session.post(
        f"{TOGETHER_API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0,
        },
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        try:
            score = int(raw)
            return max(0, min(10, score))
        except ValueError:
            return 5  # neutral fallback
```

---

### 1.4 `similarity.py` — Output divergence measurement

Two modes: Standard (trigram cosine) and Semantic (embedding cosine + z-score judge pass).

```python
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
```

---

### 1.5 `shapley.py` — Monte Carlo Shapley engine

This is the core. Ports the Shapley attribution from the web tool to Python, operating at phrase level.

```python
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
    Preserves original phrase order. Omits inactive phrases (Option B from PRD).
    For XML-tagged phrases, preserves the tag wrapper on reconstruction.
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
            full_prompt = f"{prompt}\n\n{user_input}" if user_input else prompt
            current_output = await generate(full_prompt, session=session)
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
            f"{empty_prompt_text}\n\n{test_inputs[0]}" if test_inputs else empty_prompt_text,
            session=session
        )

        cache = CoalitionCache()
        all_contributions: list[list[float]] = []   # shape: [test_inputs * m_samples, n_phrases]

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
        )  # word count as token proxy; replace with tiktoken if available
        total_tokens = sum(len(p.text.split()) for p in phrases)
        redundancy_fraction = len(low_saliency_phrases) / n if n > 0 else 0.0

        # Confidence: stddev of normalised scores across walks (lower = more stable)
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
```

---

### 1.6 `__init__.py` — Public SDK interface

```python
from .types import (
    SaliencyReport, SaliencyScore, CompressionResult,
    SimilarityMode, Phrase, RegionType
)
from .shapley import run_shapley
from .segmenter import segment_prompt

__all__ = [
    "run_shapley",
    "segment_prompt",
    "SaliencyReport",
    "SaliencyScore",
    "CompressionResult",
    "SimilarityMode",
    "Phrase",
    "RegionType",
]

__version__ = "0.1.0"
```

---

### 1.7 `pyproject.toml` for SDK

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "promptlens-sdk"
version = "0.1.0"
description = "Shapley-based prompt saliency analysis — Python SDK"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.9",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

---

## Part 2 — CLI Agent (`agent/`)

### 2.1 `discovery.py` — Codebase prompt discovery

Finds all LLM prompts in a Python or TypeScript codebase using AST analysis.

```python
import ast
import os
import re
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class DiscoveredPrompt:
    file: str
    line: int
    framework: str
    origin: str             # "literal" | "variable" | "file" | "unknown"
    prompt_text: Optional[str]   # None if origin is "unknown" (e.g., DB-fetched)
    origin_file: Optional[str]   # if origin == "file"
    estimated_tokens: int

# API call signatures to detect — extend as needed
FRAMEWORK_SIGNATURES = {
    "openai":     ["openai.chat.completions.create", "client.chat.completions.create"],
    "anthropic":  ["client.messages.create", "anthropic.messages.create"],
    "langchain":  ["ChatOpenAI", "ChatAnthropic", "LLMChain", "PromptTemplate"],
    "bedrock":    ["bedrock_runtime.invoke_model", "BedrockChat"],
}


class PythonPromptVisitor(ast.NodeVisitor):
    """AST visitor that finds LLM API calls and extracts system prompt arguments."""

    def __init__(self, source_lines: list[str], file_path: str):
        self.source_lines = source_lines
        self.file_path = file_path
        self.discovered: list[DiscoveredPrompt] = []
        self._assignments: dict[str, str] = {}  # var_name -> literal value

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track simple string assignments for variable resolution."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._assignments[target.id] = node.value.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect LLM API calls and extract system prompt."""
        call_str = ast.unparse(node)

        for framework, signatures in FRAMEWORK_SIGNATURES.items():
            if any(sig in call_str for sig in signatures):
                prompt_text, origin, origin_file = self._extract_system_prompt(node)
                if prompt_text or origin != "unknown":
                    self.discovered.append(DiscoveredPrompt(
                        file=self.file_path,
                        line=node.lineno,
                        framework=framework,
                        origin=origin,
                        prompt_text=prompt_text,
                        origin_file=origin_file,
                        estimated_tokens=len(prompt_text.split()) if prompt_text else 0,
                    ))

        self.generic_visit(node)

    def _extract_system_prompt(self, node: ast.Call):
        """
        Look for system message in messages=[{"role": "system", "content": ...}].
        Returns (prompt_text, origin, origin_file).
        """
        for keyword in node.keywords:
            if keyword.arg == "messages" and isinstance(keyword.value, ast.List):
                for elt in keyword.value.elts:
                    if isinstance(elt, ast.Dict):
                        role_val = self._get_dict_value(elt, "role")
                        if role_val == "system":
                            content = elt
                            content_val = self._get_dict_value_node(elt, "content")
                            return self._resolve_value(content_val)

        return None, "unknown", None

    def _get_dict_value(self, node: ast.Dict, key: str) -> Optional[str]:
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                if isinstance(v, ast.Constant):
                    return v.value
        return None

    def _get_dict_value_node(self, node: ast.Dict, key: str):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and k.value == key:
                return v
        return None

    def _resolve_value(self, node):
        """Resolve a value node to (text, origin, origin_file)."""
        if node is None:
            return None, "unknown", None
        if isinstance(node, ast.Constant):
            return node.value, "literal", None
        if isinstance(node, ast.Name):
            if node.id in self._assignments:
                return self._assignments[node.id], "variable", None
            return None, "variable", None
        if isinstance(node, ast.Call):
            call_str = ast.unparse(node)
            # Detect open("path").read() or Path("path").read_text()
            file_match = re.search(r'["\']([^"\']+\.(txt|md|jinja2|j2))["\']', call_str)
            if file_match:
                return None, "file", file_match.group(1)
        return None, "unknown", None


def discover_prompts(repo_path: str) -> list[DiscoveredPrompt]:
    """
    Walk repo, find all Python files, run AST visitor on each.
    Returns flat list of discovered prompts across the codebase.
    """
    discovered = []
    repo = Path(repo_path)

    for py_file in repo.rglob("*.py"):
        # Skip venv, node_modules, __pycache__, test files (configurable)
        skip_dirs = {".venv", "venv", "node_modules", "__pycache__", ".git"}
        if any(part in skip_dirs for part in py_file.parts):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            lines = source.splitlines()
            visitor = PythonPromptVisitor(lines, str(py_file))
            visitor.visit(tree)

            # Resolve file-origin prompts
            for dp in visitor.discovered:
                if dp.origin == "file" and dp.origin_file:
                    prompt_path = repo / dp.origin_file
                    if prompt_path.exists():
                        dp.prompt_text = prompt_path.read_text(encoding="utf-8")
                        dp.estimated_tokens = len(dp.prompt_text.split())

            discovered.extend(visitor.discovered)
        except (SyntaxError, UnicodeDecodeError):
            continue  # skip unparseable files

    return discovered
```

---

### 2.2 `compressor.py` — Constrained LLM rewrite

Takes a SaliencyReport and produces a compressed prompt. High-saliency phrases are locked. Only low-saliency phrases are touched.

```python
import asyncio
import aiohttp
from ..sdk.python.promptlens import SaliencyReport, SaliencyScore
from ..sdk.python.promptlens.generator import generate

COMPRESSION_SYSTEM_PROMPT = """You are a prompt compression specialist.

You will receive a prompt split into phrases, each labelled KEEP or COMPRESS.

Rules:
- KEEP phrases: reproduce them EXACTLY, word for word. Do not touch them.
- COMPRESS phrases: apply ONE of these actions and label it:
  - REMOVE: if the phrase adds no unique instruction not covered by other phrases
  - MERGE: if the phrase partially overlaps with an adjacent phrase — combine them concisely
  - REWRITE: if the phrase is load-bearing but verbose — express the same instruction in fewer tokens

Output format — for each phrase on its own line:
[KEEP] <exact original text>
[REMOVE] <original> → (removed)
[MERGE with N] <original> → <merged text>
[REWRITE] <original> → <rewritten text>

Do not add explanations. Do not change KEEP phrases. Preserve meaning of all KEEP phrases exactly."""


async def compress_prompt(report: SaliencyReport) -> tuple[str, list[dict]]:
    """
    Produce a compressed prompt from a SaliencyReport.
    Returns (compressed_prompt_text, diff_list).

    diff_list entries: {phrase, action, original, result}
    """
    # Build the labelled phrase list for the LLM
    phrase_lines = []
    for score in report.scores:
        label = "KEEP" if score.disposition == "keep" else "COMPRESS"
        phrase_lines.append(f"[{label}] {score.phrase.text}")

    user_message = "Compress the following prompt phrases:\n\n" + "\n".join(phrase_lines)

    response = await generate(
        prompt=user_message,
        system_prompt=COMPRESSION_SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=2048,
    )

    compressed_parts, diff = _parse_compression_response(response, report.scores)
    compressed_prompt = " ".join(compressed_parts)
    return compressed_prompt, diff


def _parse_compression_response(
    response: str,
    scores: list[SaliencyScore],
) -> tuple[list[str], list[dict]]:
    """
    Parse the labelled compression output back into a phrase list and diff.
    Falls back to original phrase text if parsing fails for a given line.
    """
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    compressed_parts = []
    diff = []

    for i, line in enumerate(lines):
        original = scores[i].phrase.text if i < len(scores) else ""

        if line.startswith("[KEEP]"):
            text = line[6:].strip()
            compressed_parts.append(text)
            diff.append({"phrase": i, "action": "keep", "original": original, "result": text})

        elif line.startswith("[REMOVE]"):
            diff.append({"phrase": i, "action": "remove", "original": original, "result": ""})
            # Nothing added to compressed_parts

        elif line.startswith("[MERGE"):
            arrow = line.find("→")
            result = line[arrow+1:].strip() if arrow != -1 else original
            compressed_parts.append(result)
            diff.append({"phrase": i, "action": "merge", "original": original, "result": result})

        elif line.startswith("[REWRITE]"):
            arrow = line.find("→")
            result = line[arrow+1:].strip() if arrow != -1 else original
            compressed_parts.append(result)
            diff.append({"phrase": i, "action": "rewrite", "original": original, "result": result})

        else:
            # Unparseable line — keep original as fallback
            compressed_parts.append(original)
            diff.append({"phrase": i, "action": "keep", "original": original, "result": original})

    return compressed_parts, diff
```

---

### 2.3 `validator.py` — Compression validation loop

Validates the compressed prompt against the same test inputs used in Shapley analysis.

```python
import asyncio
import aiohttp
from ..sdk.python.promptlens import SaliencyReport, SimilarityMode
from ..sdk.python.promptlens.generator import generate
from ..sdk.python.promptlens.similarity import compute_divergences

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
            # Find first removed/rewritten phrase in the failing set and restore it
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

            # Rebuild compressed prompt from updated diff
            current_compressed = " ".join(
                e["result"] for e in current_diff if e["result"]
            )

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
    and indices of phrases whose removal caused it.
    """
    divergences = []
    for user_input in test_inputs:
        original_output = await generate(f"{original_prompt}\n\n{user_input}", session=session)
        compressed_output = await generate(f"{compressed_prompt}\n\n{user_input}", session=session)
        divs = await compute_divergences(original_output, [compressed_output], mode=mode, session=session)
        divergences.append(divs[0])

    worst = max(divergences)
    failing = [i for i, d in enumerate(divergences) if d > threshold]
    return worst, failing
```

---

### 2.4 `reporter.py` — Terminal output formatting

Rich terminal output for the CLI. No external Rich dependency — uses ANSI codes directly.

```python
from ..sdk.python.promptlens import SaliencyReport, CompressionResult

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
ORANGE = "\033[93m"
GREEN  = "\033[92m"
BLUE   = "\033[94m"
GRAY   = "\033[90m"
CYAN   = "\033[96m"


def _bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    color = GREEN if score < 0.15 else (ORANGE if score < 0.5 else RED)
    return color + "█" * filled + GRAY + "░" * (width - filled) + RESET


def print_saliency_report(report: SaliencyReport, file: str = "") -> None:
    print()
    if file:
        print(f"{BOLD}📋 PromptLens Analysis — {file}{RESET}")
    print(f"{GRAY}{'─' * 60}{RESET}")
    print(f"  Phrases analysed : {BOLD}{len(report.phrases)}{RESET}")
    print(f"  Token estimate   : {BOLD}{report.token_count}{RESET}")
    print(f"  Test inputs used : {BOLD}{report.test_inputs_used}{RESET}")
    print(f"  Confidence       : {BOLD}{report.confidence:.2f}{RESET}")
    print(f"  Est. redundancy  : {BOLD}{report.redundancy_fraction * 100:.0f}%{RESET}")
    print()

    print(f"  {BOLD}{'PHRASE':<50} {'SCORE':>6}  {'IMPACT'}{RESET}")
    print(f"  {GRAY}{'─' * 50} {'─' * 6}  {'─' * 22}{RESET}")

    for score in sorted(report.scores, key=lambda s: s.score, reverse=True):
        phrase_preview = score.phrase.text[:48] + ".." if len(score.phrase.text) > 48 else score.phrase.text
        score_color = GREEN if score.score < 0.15 else (ORANGE if score.score < 0.5 else RED)
        label = f"  {phrase_preview:<50} {score_color}{score.score:>6.2f}{RESET}  {_bar(score.score)}"
        print(label)

    print()
    print(f"  {BOLD}Candidates for compression:{RESET} "
          f"{RED}{len([s for s in report.scores if s.disposition == 'remove'])} phrases{RESET} "
          f"/ {report.compression_candidate_tokens} tokens")
    print()


def print_audit_summary(discoveries: list, reports: dict) -> None:
    total_tokens = sum(r.token_count for r in reports.values())
    candidate_tokens = sum(r.compression_candidate_tokens for r in reports.values())

    print()
    print(f"{BOLD}{CYAN}🔍 PromptLens Agent — Audit Complete{RESET}")
    print(f"{GRAY}{'═' * 60}{RESET}")
    print(f"  Prompts found    : {BOLD}{len(discoveries)}{RESET}")
    print(f"  Total tokens     : {BOLD}{total_tokens:,}{RESET}")
    print(f"  Candidate tokens : {RED}{BOLD}{candidate_tokens:,}{RESET} ({candidate_tokens/total_tokens*100:.0f}% of total)")
    print()

    print(f"  {BOLD}{'FILE':<45} {'TOKENS':>7}  {'REDUNDANCY':>10}{RESET}")
    print(f"  {GRAY}{'─' * 45} {'─' * 7}  {'─' * 10}{RESET}")
    for path, report in sorted(reports.items(), key=lambda x: x[1].redundancy_fraction, reverse=True):
        fname = path[-43:] if len(path) > 43 else path
        pct = f"{report.redundancy_fraction * 100:.0f}%"
        color = RED if report.redundancy_fraction > 0.4 else (ORANGE if report.redundancy_fraction > 0.2 else GREEN)
        print(f"  {fname:<45} {report.token_count:>7,}  {color}{pct:>10}{RESET}")

    print()
    print(f"  Run {CYAN}promptlens compress --file <path>{RESET} to compress a specific prompt.")
    print()


def print_compression_result(result: CompressionResult) -> None:
    print()
    print(f"{BOLD}{CYAN}✂  Compression Result{RESET}")
    print(f"{GRAY}{'─' * 60}{RESET}")

    status = f"{GREEN}✓ PASSED{RESET}" if result.validation_passed else f"{RED}✗ FAILED{RESET}"
    print(f"  Validation       : {status}")
    print(f"  Max divergence   : {result.worst_case_divergence:.3f}")
    print(f"  Original tokens  : {result.original_tokens:,}")
    print(f"  Compressed tokens: {result.compressed_tokens:,}")
    print(f"  Token reduction  : {BOLD}{GREEN}{result.token_delta:,} tokens "
          f"({result.token_delta / result.original_tokens * 100:.0f}%){RESET}")

    print()
    print(f"  {BOLD}Changes:{RESET}")
    for entry in result.diff:
        if entry["action"] == "keep":
            continue
        icon = {"remove": "🗑 ", "rewrite": "✏️ ", "merge": "⊕ "}.get(entry["action"], "  ")
        preview = entry["original"][:50] + ".." if len(entry["original"]) > 50 else entry["original"]
        if entry["action"] == "remove":
            print(f"  {icon} {RED}{preview}{RESET}")
        else:
            result_preview = entry["result"][:40] + ".." if len(entry["result"]) > 40 else entry["result"]
            print(f"  {icon} {ORANGE}{preview}{RESET} → {GREEN}{result_preview}{RESET}")

    print()
    print(f"  Compressed prompt written to: {CYAN}<file>.suggested{RESET}")
    print()
```

---

### 2.5 `cli.py` — Click CLI entry point

Three verbs: `check` (fast), `audit` (full), `compress` (full + rewrite + validate).

```python
import asyncio
import click
from pathlib import Path
from .discovery import discover_prompts
from .reporter import print_saliency_report, print_audit_summary, print_compression_result
from .compressor import compress_prompt
from .validator import validate_compression
from ..sdk.python.promptlens import run_shapley, SimilarityMode, CompressionResult


def _get_mode(semantic: bool) -> SimilarityMode:
    return SimilarityMode.SEMANTIC if semantic else SimilarityMode.STANDARD


def _load_test_inputs(test_inputs_file: str | None, prompt_text: str, n: int = 10) -> list[str]:
    """Load test inputs from file, or generate synthetic ones."""
    if test_inputs_file:
        path = Path(test_inputs_file)
        if path.suffix == ".jsonl":
            import json
            lines = path.read_text().strip().splitlines()
            return [json.loads(l)["input"] for l in lines[:n]]
        else:
            return [l.strip() for l in path.read_text().splitlines() if l.strip()][:n]

    # Synthetic generation: use a simple heuristic — generate diverse inputs
    # In production: call generate() with the prompt to produce synthetic inputs
    # For now: return a placeholder that Claude Code should replace with actual generation
    return [
        "Please help me with this task.",
        "What should I do in this situation?",
        "Give me your best recommendation.",
    ]


@click.group()
def cli():
    """PromptLens Agent — Evidence-based prompt optimisation."""
    pass


@cli.command()
@click.option("--file", "-f", required=True, help="Path to prompt file to check.")
@click.option("--semantic", is_flag=True, default=False, help="Use semantic similarity (requires Together AI key).")
def check(file: str, semantic: bool):
    """
    Fast saliency check on a single prompt file. M=3 samples.
    Use before commits to catch newly introduced bloat.
    """
    prompt_text = Path(file).read_text(encoding="utf-8")
    test_inputs = _load_test_inputs(None, prompt_text, n=3)

    async def run():
        report = await run_shapley(
            prompt=prompt_text,
            test_inputs=test_inputs,
            m_samples=3,                     # fast mode
            mode=_get_mode(semantic),
        )
        print_saliency_report(report, file=file)
        if report.redundancy_fraction > 0.2:
            click.echo(f"⚠️  {report.redundancy_fraction*100:.0f}% of phrases appear low-impact.")
            click.echo(f"   Run: promptlens compress --file {file}")

    asyncio.run(run())


@cli.command()
@click.option("--repo", "-r", default=".", help="Path to repository root. Default: current directory.")
@click.option("--file", "-f", default=None, help="Analyse a specific file only.")
@click.option("--semantic", is_flag=True, default=False)
@click.option("--test-inputs", "test_inputs_file", default=None, help="Path to .jsonl or .txt file of test inputs.")
@click.option("--m-samples", default=20, help="Monte Carlo samples per test input. Default: 20.")
def audit(repo: str, file: str | None, semantic: bool, test_inputs_file: str | None, m_samples: int):
    """
    Full saliency audit. Discovers all prompts in repo (or analyses a single file).
    Outputs a compression brief per prompt.
    """
    if file:
        targets = [Path(file)]
    else:
        discoveries = discover_prompts(repo)
        targets = [
            Path(d.origin_file or d.file)
            for d in discoveries
            if d.prompt_text
        ]
        if not targets:
            click.echo("No prompts found. Is this a Python codebase with OpenAI/Anthropic calls?")
            return

    async def run():
        reports = {}
        for target in targets:
            prompt_text = target.read_text(encoding="utf-8")
            test_inputs = _load_test_inputs(test_inputs_file, prompt_text)
            report = await run_shapley(
                prompt=prompt_text,
                test_inputs=test_inputs,
                m_samples=m_samples,
                mode=_get_mode(semantic),
            )
            reports[str(target)] = report
            print_saliency_report(report, file=str(target))

        if len(reports) > 1:
            print_audit_summary(targets, reports)

    asyncio.run(run())


@cli.command()
@click.option("--file", "-f", required=True, help="Path to prompt file to compress.")
@click.option("--threshold", default=0.15, help="Max output divergence allowed. Default: 0.15.")
@click.option("--semantic", is_flag=True, default=False)
@click.option("--test-inputs", "test_inputs_file", default=None)
@click.option("--m-samples", default=20)
def compress(file: str, threshold: float, semantic: bool, test_inputs_file: str | None, m_samples: int):
    """
    Full compression pipeline: analyse → rewrite → validate → write .suggested file.
    Does not overwrite the original. Developer reviews and accepts manually.
    """
    prompt_path = Path(file)
    prompt_text = prompt_path.read_text(encoding="utf-8")
    test_inputs = _load_test_inputs(test_inputs_file, prompt_text)
    mode = _get_mode(semantic)

    async def run():
        # Step 1: Shapley analysis
        click.echo(f"⚙  Running Shapley analysis (M={m_samples})...")
        report = await run_shapley(
            prompt=prompt_text,
            test_inputs=test_inputs,
            m_samples=m_samples,
            mode=mode,
        )
        print_saliency_report(report, file=file)

        # Step 2: Constrained compression
        click.echo("✂  Compressing low-saliency phrases...")
        compressed, diff = await compress_prompt(report)

        # Step 3: Validation loop
        click.echo(f"🔍 Validating against {len(test_inputs)} test inputs (threshold={threshold})...")
        passed, worst_divergence, final_compressed = await validate_compression(
            original_prompt=prompt_text,
            compressed_prompt=compressed,
            test_inputs=test_inputs,
            report=report,
            diff=diff,
            threshold=threshold,
            mode=mode,
        )

        # Step 4: Write output
        original_tokens = len(prompt_text.split())
        compressed_tokens = len(final_compressed.split())
        result = CompressionResult(
            original_prompt=prompt_text,
            compressed_prompt=final_compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            token_delta=original_tokens - compressed_tokens,
            validation_passed=passed,
            worst_case_divergence=worst_divergence,
            saliency_report=report,
            diff=diff,
        )

        suggested_path = prompt_path.with_suffix(prompt_path.suffix + ".suggested")
        suggested_path.write_text(final_compressed, encoding="utf-8")

        print_compression_result(result)

        if not passed:
            click.echo("⚠️  Validation did not fully pass. Review .suggested file carefully before adopting.")

    asyncio.run(run())


def main():
    cli()
```

---

### 2.6 `pyproject.toml` for Agent

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "promptlens-agent"
version = "0.1.0"
description = "PromptLens Agent — evidence-based prompt optimisation CLI"
requires-python = ">=3.11"
dependencies = [
    "promptlens-sdk>=0.1.0",
    "click>=8.1",
    "aiohttp>=3.9",
    "langgraph>=0.1",
]

[project.scripts]
promptlens = "promptlens_agent.cli:main"

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]
```

---

## Part 3 — Installation and Usage

### Install

```bash
# From repo root
pip install -e sdk/python
pip install -e agent

# Set API key
export TOGETHER_API_KEY=your_key_here
```

### Commands

```bash
# Fast check on a single prompt file (M=3, warn-only)
promptlens check --file src/prompts/system.txt

# Full audit of entire repo
promptlens audit --repo .

# Full audit of single file with semantic similarity
promptlens audit --file src/prompts/system.txt --semantic

# Compress with custom threshold and test inputs
promptlens compress \
  --file src/prompts/system.txt \
  --threshold 0.12 \
  --test-inputs tests/sample_inputs.jsonl \
  --semantic
```

### Output files

`promptlens compress` writes `<original_file>.suggested` alongside the original. Never overwrites the original. Developer diffs and adopts manually:

```bash
diff src/prompts/system.txt src/prompts/system.txt.suggested
```

---

## Part 4 — Implementation Order for Claude Code

Do these in order. Each step is independently testable.

**Step 1** — Create monorepo structure. Move existing `web/` files into `web/`. Create `sdk/python/promptlens/` and `agent/promptlens_agent/` directories.

**Step 2** — Implement `types.py`. No dependencies. Run: `python -c "from promptlens.types import SaliencyReport; print('ok')"`.

**Step 3** — Implement `segmenter.py`. Write `tests/test_segmenter.py` with four test cases: plain text, JSON, bullets, XML. Run `pytest sdk/python/tests/test_segmenter.py`.

**Step 4** — Implement `generator.py`. Test with a live Together AI key: `python -c "import asyncio; from promptlens.generator import generate; print(asyncio.run(generate('hello')))"`.

**Step 5** — Implement `similarity.py`. Test standard mode (no API key needed): trigram cosine between two similar strings should return < 0.3.

**Step 6** — Implement `shapley.py`. Test with a short 3-phrase prompt and M=3. Verify scores sum is non-zero and high-saliency phrase gets highest score.

**Step 7** — Implement `discovery.py`. Test against a sample Python file that contains an `openai.chat.completions.create` call with a system message.

**Step 8** — Implement `compressor.py` and `validator.py`.

**Step 9** — Implement `reporter.py`. Test terminal output manually: run `promptlens check` and verify formatting.

**Step 10** — Wire up `cli.py`. Test all three commands end-to-end.

---

## Part 5 — What NOT to build in this pass

- GitHub Action (Phase 4 — future)
- Pre-commit hook (future)
- TypeScript SDK (future)
- Synthetic test input generation using LLM calls (use the 3-line placeholder in `_load_test_inputs` for now)
- tiktoken integration for accurate token counts (word-split estimate is sufficient for v1)
- Web UI changes (do not touch `web/` directory)
