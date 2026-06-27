# PromptLens

**Find the dead weight in your LLM prompts — and cut it safely.**

PromptLens uses Shapley-value attribution to score every phrase in your system prompts by how much it actually changes model output. Phrases that score near zero are candidates for removal. Phrases that score high are load-bearing — don't touch them.

The CLI agent finds all your prompts automatically, audits them, and can compress them while running empirical validation to make sure the behaviour doesn't change.

**Live web tool (paste-and-go):** https://rohityalavarthy.github.io/PromptLens

---

## Quickstart

```bash
# 1. Install
pip install -e sdk/python
pip install -e agent

# 2. Set your API key (Together AI — free $1 credit at together.ai)
export TOGETHER_API_KEY=your_key_here

# 3. Check a single prompt file
promptlens check --file src/prompts/system.txt

# 4. Audit your whole repo
promptlens audit --repo .

# 5. Compress a bloated prompt
promptlens compress --file src/prompts/system.txt
```

Requires Python 3.11+.

---

## CLI Agent

### `check` — fast pre-commit scan

Runs a quick saliency analysis on a single prompt file and warns you if more than 20% looks redundant. Fast enough to run before every commit.

```bash
promptlens check --file src/prompts/system.txt
```

**Example output:**

```
📋 PromptLens Analysis — src/prompts/system.txt
────────────────────────────────────────────────────
  Phrases analysed : 5
  Token estimate   : 94
  Test inputs used : 3
  Confidence       : 0.88
  Est. redundancy  : 31%

  PHRASE                                       SCORE  IMPACT
  ──────────────────────────────────────────── ────── ─────────────────────────
  Always respond concisely.                     0.91  ████████████████████
  Use bullet points when listing items.         0.54  ████████████
  Format your answer in Markdown.               0.42  █████████
  Respond in the user's language.               0.18  ████
  You are a helpful assistant.                  0.08  █

  ⚠  Redundancy warning: 31% of tokens score below threshold (0.15)
  Candidates for compression: 2 phrases / 29 tokens
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--file, -f` | required | Path to the prompt file |
| `--saliency-threshold` | `0.15` | Phrases below this score are flagged |
| `--semantic` | off | Use embedding similarity (requires Together AI key) |

---

### `audit` — full repo analysis

Walks your entire codebase, finds every LLM API call, extracts the system prompts, and runs a saliency analysis on each one. Produces a ranked summary showing which prompts have the most redundancy.

```bash
# Scan the whole repo
promptlens audit --repo .

# Scan a single file
promptlens audit --file src/api/openai_handler.py

# Use your own test inputs for more realistic scoring
promptlens audit --repo . --test-inputs tests/inputs.jsonl
```

**Example output:**

```
🔍 PromptLens Agent — Audit Complete
════════════════════════════════════════════════════
  Prompts found    : 5
  Total tokens     : 2,341
  Candidate tokens : 418 (18% of total)

  FILE                                   TOKENS  REDUNDANCY
  ─────────────────────────────────────── ────── ──────────
  src/api/openai_handler.py                  234      42%
  src/prompts/search.txt                     189      12%
  src/llm/anthropic_wrapper.py               127       8%
  src/agents/summarizer.py                    89       4%

  Run promptlens compress --file <path> to compress a specific prompt.
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--repo, -r` | `.` | Repo root to scan |
| `--file, -f` | — | Analyse a single file instead of scanning |
| `--test-inputs` | — | `.jsonl` or `.txt` file of test inputs |
| `--m-samples` | `20` | Monte Carlo walks for Shapley (higher = more precise) |
| `--saliency-threshold` | `0.15` | Score below which a phrase is a compression candidate |
| `--semantic` | off | Use embedding similarity instead of trigram |

**What it finds:**

The agent uses AST analysis to detect LLM API calls across all common frameworks:

| Framework | Detected patterns |
|---|---|
| OpenAI | `openai.chat.completions.create`, `client.chat.completions.create` |
| Anthropic | `client.messages.create`, `anthropic.messages.create` |
| LangChain | `ChatOpenAI`, `ChatAnthropic`, `LLMChain`, `PromptTemplate` |
| AWS Bedrock | `bedrock_runtime.invoke_model`, `BedrockChat` |

It resolves prompts from three sources:
- **Literal strings** — `messages=[{"role": "system", "content": "You are..."}]`
- **Variable assignments** — `SYSTEM_PROMPT = "..."` then used in the call
- **File reads** — `open("prompts/system.txt").read()` or `Path(...).read_text()`

---

### `compress` — analyse, rewrite, validate

The full pipeline: scores your prompt, sends low-saliency phrases to an LLM rewriter with explicit instructions, validates that the compressed prompt produces similar outputs, and retries if it doesn't.

```bash
# Basic compression
promptlens compress --file src/prompts/system.txt

# Tighter quality gate
promptlens compress --file src/prompts/system.txt --threshold 0.10

# Semantic similarity + your own test inputs
promptlens compress --file src/prompts/system.txt --semantic --test-inputs tests/inputs.jsonl

# Review output then optionally apply in-place
promptlens compress --file src/prompts/system.txt --apply
```

**How it works, step by step:**

1. **Score** — Runs Shapley analysis to get a 0–1 impact score for every phrase.
2. **Label** — Phrases below the saliency threshold get labeled `[COMPRESS]`; others get `[KEEP]`. Each label includes the exact score so the rewriter knows how aggressively to act.
3. **Rewrite** — Sends the labeled prompt to a Qwen rewriter with a score-based decision table:
   - `0.00–0.05` → REMOVE (if covered elsewhere in the prompt)
   - `0.05–0.10` → REMOVE, MERGE, or REWRITE
   - `0.10–threshold` → MERGE or REWRITE only
   - `≥ threshold` → PARAPHRASE only (no structural change)
4. **Validate** — Runs both the original and compressed prompts against all test inputs, measures output divergence. Computes a verdict:
   - **PASS** — safe to adopt
   - **MARGINAL** — spot-check outputs before committing
   - **REVIEW** — differences are significant; read carefully
   - **FAIL** — do not apply without thorough manual review
5. **Retry** — If validation fails, the agent reinstates the modified phrase with the highest Shapley score and re-validates. Up to 3 retries.
6. **Write** — Result is written to `<file>.suggested`. Your original is never touched unless you use `--apply` and confirm.

**Example output:**

```
✂  Compression Result
────────────────────────────────────────────────────
  Validation       : ✓ PASS
  Max divergence   : 0.087
  Original tokens  : 127
  Compressed tokens: 89
  Token reduction  : 38 tokens (30%)

  Changes:
    🗑  Redundant phrase covered elsewhere
    ✏️  You are extremely helpful and thorough in your responses. → Be thorough.
    ↺  Please assist users in a polite and professional manner. → Be polite and professional.

  Compressed prompt written to: src/prompts/system.txt.suggested
```

**Review the diff before applying:**

```bash
diff src/prompts/system.txt src/prompts/system.txt.suggested
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--file, -f` | required | Prompt file to compress |
| `--threshold` | `0.15` | Max output divergence allowed during validation |
| `--saliency-threshold` | auto | Phrases below this get labeled COMPRESS. Defaults to `min(threshold, 0.50)` |
| `--test-inputs` | — | `.jsonl` (`{"input": "..."}`) or `.txt` (one per line) |
| `--m-samples` | `20` | Monte Carlo walks for Shapley |
| `--semantic` | off | Use embeddings for divergence measurement |
| `--apply` | off | After writing `.suggested`, prompt to overwrite original in-place |

**Tip on test inputs:** If you don't provide a `--test-inputs` file, the agent falls back to three generic inputs. The saliency scores will be less accurate. For production prompts, always pass representative real-world inputs — your results will be much sharper.

---

## Test input file formats

**JSONL** (one JSON object per line):
```jsonl
{"input": "Summarize the following article in three sentences."}
{"input": "What is the capital of France?"}
{"input": "Write a Python function that sorts a list."}
```

**Plain text** (one input per line):
```
Summarize the following article in three sentences.
What is the capital of France?
Write a Python function that sorts a list.
```

---

## Similarity modes

Both `check`, `audit`, and `compress` support two ways to measure output divergence:

**Standard (default):** Character trigram cosine distance. No extra API calls. Fast and works offline. Good for prompts where output wording is important.

**Semantic (`--semantic`):** Embedding cosine distance via `nomic-ai/nomic-embed-text-v1.5` (Together AI). Captures meaning-level change — rewording the same idea doesn't register as divergence. Use this when you care about semantic equivalence, not exact phrasing.

---

## M-samples and speed

All three commands accept `--m-samples` to control the Shapley sampling budget.

| Value | Use case | Est. API calls (N=10 phrases) |
|---|---|---|
| `3` | Pre-commit, instant feedback | ~6–10 |
| `20` | Default — balanced quality | ~20–40 |
| `50` | High-confidence audit | ~50–80 |

For prompts with N ≤ 4 phrases, PromptLens computes **exact** Shapley values (all 2^N coalitions) regardless of `--m-samples`. Coalition outputs are cached — concurrent walks that hit the same subset share a single in-flight API call.

All generation calls use `temperature: 0.0` — determinism is required for stable divergence measurement.

---

## Python SDK

If you want to run Shapley attribution from your own code:

```bash
pip install -e sdk/python
```

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

**`SaliencyReport` fields:**
- `scores` — list of `SaliencyScore`: `.score` (0–1), `.raw_shapley`, `.disposition` (`keep`/`remove`)
- `phrases` — segmented phrase list
- `token_count`, `redundancy_fraction`, `compression_candidate_tokens`
- `confidence`, `m_samples`, `test_inputs_used`

---

## Web tool

Paste a prompt directly in the browser and get it back colour-coded by impact. No install, no API key to configure in a terminal.

**Run at:** https://rohityalavarthy.github.io/PromptLens

**Run locally:**
```bash
cd web && python3 -m http.server 8080
# open http://localhost:8080
```

### Providers

| Provider | Saliency | Semantic similarity | Notes |
|---|---|---|---|
| Groq | ✓ | — | Free tier, no credit card. Fastest for saliency. |
| Together AI | ✓ | ✓ (nomic-embed) | $1 free credit. Enables embedding-based similarity. |
| OpenAI | ✓ | ✓ (text-embedding-3-small) | GPT-4o mini for analysis, embeddings for semantic mode. |
| Anthropic | ✓ | — | Claude Haiku. No embedding endpoint; falls back to trigram similarity. |

Keys are stored in `localStorage` — they never leave your browser.

### Compress

After running an analysis, a **Compress** button appears in the results panel. It removes all phrases scoring below your chosen threshold and runs a single LLM pass to fix any structural issues left behind (broken JSON, orphaned XML tags, incomplete bullet lists) — without rephrasing or adding back content.

**Flow:**
1. Run analysis → phrases are scored and colour-coded
2. Click **Compress** → set the **Remove below** threshold (default `< 0.15`)
3. A structural cleanup LLM call patches formatting only
4. Review the diff — every removed phrase is listed
5. Adjust the threshold and click **Recompress** to iterate
6. Click **Apply** to copy the compressed prompt back into the editor, or **Discard** to cancel

Atomic phrases (code blocks, XML-tagged sections) are never removed regardless of score.

### Other features

- **Heatmap / Table view** — toggle between the colour-coded heatmap and a sortable score table
- **Threshold filter** — slider to dim phrases below a score, focusing the heatmap on high-impact content
- **Copy / Export** — copy results as a markdown table, or export full JSON (phrases, scores, settings, baseline response)
- **Share** — encodes the current analysis state into a URL fragment; anyone with the link can restore the exact view
- **Run history** — last 8 analyses stored in `localStorage`; accessible via the History dropdown next to the run button
- **Auto-save** — prompt text is saved to `localStorage` on every keystroke and restored on reload

---

## How attribution works

Standard perturbation methods (leave-one-out, ablation) test phrases in isolation and miss interactions. A phrase can look useless alone but be essential when combined with others.

Shapley values fix this. The Shapley value for a phrase is its **average marginal contribution across all possible coalitions of the other phrases** — the only attribution method satisfying all four fairness axioms when features interact (efficiency, symmetry, dummy, additivity).

For N ≤ 4 phrases: exact computation (all 2^N subsets). For N > 4: Monte Carlo sampling with M random coalition walks, concurrency-capped at 5. Coalition cache deduplicates repeated subset calls across walks.

---

## Development

```bash
# SDK — 19 tests, all offline
cd sdk/python && pytest tests/ -v

# Agent — offline tests
cd agent && pytest tests/ -v
```

No live API calls in the test suite.

---

## License

MIT
