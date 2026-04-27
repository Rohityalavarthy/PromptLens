# PromptLens

Shapley-value attribution for LLM prompts. Tells you which phrases in your system prompt are load-bearing and which are dead weight — with empirical evidence, not guesswork.

**Live web tool:** https://rohityalavarthy.github.io/PromptLens

---

## What's in this repo

```
PromptLens/
├── web/          Browser-based saliency debugger (no install, paste and run)
├── sdk/python/   Python SDK — core Shapley engine, importable in your own code
└── agent/        CLI tool — discovers prompts in a codebase, audits and compresses them
```

All three share the same attribution engine. The web tool is the interactive version; the SDK and CLI are for integrating into development workflows.

---

## Web Tool

Paste a system prompt, run the analysis, get your prompt back colour-coded by how much each phrase actually influences the model output. Blue = low impact, red = high impact.

**Run directly at:** https://rohityalavarthy.github.io/PromptLens

**Run locally:**
```bash
cd web
python3 -m http.server 8080
# open http://localhost:8080
```

No build step. No dependencies. Just three files.

### How it works

1. **Segmentation** — The prompt is classified into structural regions (plain prose, JSON, XML, bullets, code blocks, markdown) then split by region-appropriate rules. Plain text splits at sentence/clause boundaries; JSON splits per top-level key; XML per tag pair; bullets per item; code blocks and headers stay atomic.

2. **Shapley attribution** — Each phrase's score is its average marginal contribution across all possible coalitions of other phrases. This correctly handles phrase interactions (a phrase can look useless in isolation but be essential in combination). For N ≤ 4 phrases: exact Shapley. For N > 4: Monte Carlo sampling with M random coalition walks (default M=20, concurrency cap 5). Coalition outputs are cached so repeated subsets don't duplicate API calls.

3. **Divergence measurement** — Standard mode: character trigram cosine distance, no extra API calls. Semantic mode: embedding cosine via `nomic-ai/nomic-embed-text-v1.5` (Together AI key required). Semantic mode captures meaning-level change rather than surface rewording.

4. **Rendering** — Raw Shapley scores are min-max normalised, then each phrase is rendered as an inline span coloured across a 5-stop gradient. Hover any phrase for its exact impact percentage.

### API keys

Keys are stored in `localStorage` — they never leave your browser.

| Provider | Model | Notes |
|---|---|---|
| Groq | `llama-3.3-70b-versatile` | Recommended for Standard mode. Free tier, no credit card. |
| Together AI | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Required for Semantic mode. $1 free credit. |

---

## Python SDK

```bash
pip install -e sdk/python
```

Requires Python 3.11+. Single dependency: `aiohttp`.

```python
import asyncio
from promptlens import run_shapley, SimilarityMode

report = asyncio.run(run_shapley(
    prompt="You are a helpful assistant. Always respond concisely. Use bullet points when listing items.",
    test_inputs=["What are the benefits of exercise?"],
    m_samples=20,
    mode=SimilarityMode.STANDARD,
))

for score in sorted(report.scores, key=lambda s: s.score, reverse=True):
    print(f"{score.score:.2f}  {score.phrase.text}")
```

```
0.91  Always respond concisely.
0.54  Use bullet points when listing items.
0.12  You are a helpful assistant.
```

### API

```python
run_shapley(
    prompt: str,
    test_inputs: list[str],
    m_samples: int = 20,           # 3=fast, 20=balanced, 50=precise
    mode: SimilarityMode = STANDARD,
    low_saliency_threshold: float = 0.15,
) -> SaliencyReport
```

```python
segment_prompt(prompt: str) -> list[Phrase]
```

**`SaliencyReport`** fields:
- `phrases` — segmented phrase list
- `scores` — `SaliencyScore` per phrase: `.score` (0–1 normalised), `.raw_shapley`, `.disposition` (keep/remove)
- `token_count`, `redundancy_fraction`, `compression_candidate_tokens`
- `confidence`, `m_samples`, `test_inputs_used`

Set `TOGETHER_API_KEY` in your environment. The SDK calls Together AI for all LLM and embedding calls.

---

## CLI Agent

Finds LLM prompts in a Python codebase via AST analysis, runs Shapley attribution on each, and optionally compresses bloated prompts with empirical validation.

### Install

```bash
pip install -e sdk/python    # install SDK first
pip install -e agent
export TOGETHER_API_KEY=your_key_here
```

### Commands

#### `check` — fast pre-commit scan

```bash
promptlens check --file src/prompts/system.txt
```

M=3 samples. Prints a phrase-level saliency table and warns if redundancy exceeds 20%. Fast enough to run before every commit.

#### `audit` — full repo audit

```bash
# Scan entire repo for LLM API calls
promptlens audit --repo .

# Single file, full analysis
promptlens audit --file src/prompts/system.txt --m-samples 20

# With your own test inputs
promptlens audit --file src/prompts/system.txt --test-inputs tests/inputs.jsonl

# Semantic similarity mode
promptlens audit --file src/prompts/system.txt --semantic
```

Discovers prompts by walking `.py` files and detecting `openai.chat.completions.create`, `anthropic.messages.create`, LangChain, and Bedrock call signatures. Extracts system prompts from literal strings, variable assignments, and file reads.

#### `compress` — analyse, rewrite, validate

```bash
promptlens compress --file src/prompts/system.txt
promptlens compress --file src/prompts/system.txt --threshold 0.12 --semantic
```

Full pipeline:
1. Run Shapley analysis to identify low-saliency phrases
2. Label each phrase KEEP or COMPRESS, send to LLM rewriter
3. Rewriter applies REMOVE / MERGE / REWRITE per phrase, keeping high-saliency phrases word-for-word
4. Validate compressed prompt against original: run both against all test inputs, compare divergence
5. If divergence exceeds threshold, reinstate the offending phrase and retry (up to 3 times)
6. Write result to `<file>.suggested` — original is never overwritten

```bash
diff src/prompts/system.txt src/prompts/system.txt.suggested
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--threshold` | `0.15` | Max output divergence allowed |
| `--m-samples` | `20` | Monte Carlo walks for Shapley |
| `--semantic` | off | Use embedding similarity instead of trigram |
| `--test-inputs` | — | `.jsonl` (`{"input": "..."}`) or `.txt` (one per line) |

If no `--test-inputs` file is provided, the agent uses three generic fallback inputs. For production use, provide representative inputs — the quality of the saliency signal depends on having realistic test cases.

---

## How attribution works

The three classic perturbation methods (Leave-One-Out, Perturbation, Paraphrase) all test phrases in isolation, which misses interactions. A phrase can look low-impact alone but be essential in combination with others.

Shapley values fix this. The Shapley value for a phrase is its **average marginal contribution across all possible coalitions of the other phrases** — the only attribution method satisfying all fairness axioms when features interact.

**API call budget:**

| Prompt size | Mode | Est. model calls |
|---|---|---|
| N ≤ 4 | Exact (all N! permutations) | ≤ 16 (with cache) |
| N = 10, M = 20 | Monte Carlo | ~20–40 (with cache hits) |
| N = 10, M = 50 | Monte Carlo | ~50–80 (with cache hits) |

Coalition outputs are cached by subset key. Concurrent walks that hit the same coalition share a single in-flight API call.

`temperature: 0.0` on all generation calls — determinism is essential for stable divergence measurement.

---

## Development

```bash
# SDK tests
cd sdk/python && pytest tests/ -v

# Agent tests
cd agent && pytest tests/ -v
```

Tests cover: segmenter (plain text, JSON, bullets, XML, code blocks, mixed), Shapley pure logic (prompt reconstruction, coalition cache), trigram similarity, and AST prompt discovery.

No live API calls in the test suite — all tests are offline.

---

## Roadmap

- [ ] Export saliency map as image
- [ ] Side-by-side diff of two prompt variants
- [ ] Batch mode across multiple test inputs with aggregated scores
- [ ] OpenAI and Anthropic provider support (web tool)
- [ ] GitHub Action for CI integration
- [ ] Pre-commit hook

---

## License

MIT
