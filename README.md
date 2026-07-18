# PromptLens

**Evidence-based prompt engineering for production LLM applications.**

PromptLens uses [Shapley-value attribution](https://en.wikipedia.org/wiki/Shapley_value) to score every phrase in your system prompts by how much it actually changes model output. Low-scoring phrases are dead weight. High-scoring phrases are load-bearing. The CLI agent finds prompts across your codebase, audits them, compresses them safely, and validates that behavior doesn't change — with full explainability at every step.

```bash
pip install -e sdk/python && pip install -e agent
export TOGETHER_API_KEY=your_key_here   # free $1 credit at together.ai
promptlens audit --repo . --format json
```

**Live web tool (no install):** https://rohityalavarthy.github.io/PromptLens

---

## Table of Contents

- [Why PromptLens](#why-promptlens)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Multi-Provider Support](#multi-provider-support)
- [CI Integration](#ci-integration)
- [Python SDK](#python-sdk)
- [How Attribution Works](#how-attribution-works)
- [Web Tool](#web-tool)
- [Development](#development)
- [License](#license)

---

## Why PromptLens

Most prompt optimization is guesswork. You shorten a prompt, eyeball a few outputs, and hope nothing breaks. PromptLens replaces that with measurement:

- **Quantified impact** — Every phrase gets a 0–1 score based on its actual contribution to model output across multiple test inputs.
- **Safe compression** — Automated rewriting with empirical validation. If the compressed prompt changes behavior beyond your threshold, it's rejected.
- **Full explainability** — JSON/SARIF reports with per-phrase rationale, validation evidence, and cost estimates. Suitable for compliance and audit trails.
- **CI-ready** — Run in GitHub Actions, get SARIF results in your code scanning dashboard, fail PRs that introduce bloated prompts.

---

## Installation

**Requirements:** Python 3.11+

```bash
# From source
git clone https://github.com/Rohityalavarthy/PromptLens.git
cd PromptLens
pip install -e sdk/python
pip install -e agent

# Verify
promptlens --help
```

**API key** (one of):
```bash
export TOGETHER_API_KEY=...    # Together AI (default provider)
export OPENAI_API_KEY=...      # OpenAI
export ANTHROPIC_API_KEY=...   # Anthropic (generation only, no semantic mode)
```

---

## Quick Start

```bash
# Check a single prompt file for redundancy
promptlens check --file prompts/system.txt

# Audit all prompts in your codebase
promptlens audit --repo .

# Compress a prompt (with validation)
promptlens compress --file prompts/system.txt

# Estimate cost before running
promptlens compress --file prompts/system.txt --dry-run

# Batch compress all discovered prompts
promptlens compress --batch --repo .

# CI mode: JSON output, non-interactive, proper exit codes
promptlens audit --repo . --ci
```

---

## CLI Reference

### `promptlens check` — Single prompt analysis

Runs Shapley analysis on one prompt and reports saliency scores. Fast enough for pre-commit hooks.

```bash
promptlens check --file src/prompts/system.txt
```

**Example output:**
```
📋 PromptLens Analysis — src/prompts/system.txt
────────────────────────────────────────────────────
  Phrases analysed : 5
  Token estimate   : 94
  Confidence       : 0.88
  Est. redundancy  : 31%

  PHRASE                                       SCORE  IMPACT
  ──────────────────────────────────────────── ────── ──────────────────
  Always respond concisely.                     0.91  ████████████████████
  Use bullet points when listing items.         0.54  ████████████
  Format your answer in Markdown.               0.42  █████████
  Respond in the user's language.               0.18  ████
  You are a helpful assistant.                  0.08  █

  ⚠  Redundancy warning: 31% of tokens score below threshold (0.15)
```

| Flag | Default | Description |
|------|---------|-------------|
| `--file, -f` | required | Path to the prompt file |
| `--saliency-threshold` | `0.15` | Phrases below this score are flagged |
| `--m-samples` | `20` | Monte Carlo walks (higher = more precise) |
| `--semantic` | off | Use embedding similarity instead of trigram |
| `--format` | `terminal` | Output format: `terminal`, `json`, or `sarif` |
| `--ci` | off | Non-interactive mode (JSON output, no prompts) |

**Exit codes:** `0` = clean, `1` = redundancy > 20%, `2` = error.

---

### `promptlens audit` — Full codebase scan

Discovers all LLM prompts across your codebase and runs saliency analysis on each.

```bash
promptlens audit --repo .
promptlens audit --repo . --format sarif > results.sarif
```

**What it discovers:**

| Language | Patterns |
|----------|----------|
| Python | `messages=[{"role": "system", ...}]`, `system="..."` (Anthropic), `SystemMessage(content=...)` (LangChain), f-strings, `.format()`, concatenation, variable assignments |
| TypeScript/JS | `role: "system"` in messages arrays, `system:` kwargs, template literals, `const SYSTEM_PROMPT = ...` |
| YAML/Config | Keys named `system_prompt`, `prompt`, `template`, `system_message`, `instructions` |
| Template files | `.jinja2`, `.j2`, `.prompt` files in `prompts/` or `templates/` directories |

| Flag | Default | Description |
|------|---------|-------------|
| `--repo, -r` | `.` | Repository root to scan |
| `--file, -f` | — | Analyse a single file instead |
| `--test-inputs` | — | `.jsonl` or `.txt` file of test inputs |
| `--m-samples` | `20` | Monte Carlo walks for Shapley |
| `--saliency-threshold` | `0.15` | Score below which a phrase is flagged |
| `--semantic` | off | Use embedding similarity |
| `--format` | `terminal` | Output: `terminal`, `json`, or `sarif` |
| `--ci` | off | Non-interactive CI mode |

---

### `promptlens compress` — Analyse, rewrite, validate

The full pipeline: score → rewrite → validate → retry. Produces a `.suggested` file with the compressed prompt.

```bash
# Basic compression
promptlens compress --file prompts/system.txt

# Tighter quality gate
promptlens compress --file prompts/system.txt --threshold 0.10

# With your own test inputs for better accuracy
promptlens compress --file prompts/system.txt --test-inputs tests/inputs.jsonl

# Apply in-place after review
promptlens compress --file prompts/system.txt --apply

# Batch: compress all discovered prompts
promptlens compress --batch --repo .

# Estimate cost without running
promptlens compress --file prompts/system.txt --dry-run
```

**How it works:**

1. **Score** — Shapley analysis assigns a 0–1 impact score to every phrase.
2. **Label** — Phrases below the saliency threshold are labeled `[COMPRESS]` with their score.
3. **Rewrite** — An LLM rewrites labeled phrases using a score-based decision table:
   - `0.00–0.05` → REMOVE (if covered elsewhere)
   - `0.05–0.10` → REMOVE, MERGE, or REWRITE
   - `0.10–threshold` → MERGE or REWRITE only
   - `≥ threshold` → PARAPHRASE only (no structural change)
4. **Validate** — Runs both prompts against test inputs and measures output divergence.
5. **Retry** — If validation fails, reinstates the highest-scoring modified phrase and re-validates (up to 3 retries).
6. **Write** — Result written to `<file>.suggested`. Original is never touched unless you use `--apply`.

**Validation verdicts:**
- **PASS** — Safe to adopt.
- **MARGINAL** — Spot-check outputs before committing.
- **REVIEW** — Differences are significant; read carefully.
- **FAIL** — Do not apply without thorough manual review.

| Flag | Default | Description |
|------|---------|-------------|
| `--file, -f` | — | Prompt file to compress |
| `--batch` | off | Compress all discovered prompts |
| `--repo, -r` | `.` | Repository root (for `--batch`) |
| `--threshold` | `0.15` | Max output divergence allowed |
| `--saliency-threshold` | auto | Phrases below this get labeled COMPRESS |
| `--test-inputs` | — | `.jsonl` or `.txt` test inputs |
| `--m-samples` | `20` | Monte Carlo walks |
| `--semantic` | off | Use embeddings for divergence |
| `--apply` | off | Overwrite original after confirmation |
| `--dry-run` | off | Estimate cost without running |
| `--format` | `terminal` | Output: `terminal`, `json`, or `sarif` |
| `--ci` | off | Non-interactive CI mode |

**Exit codes:** `0` = PASS/MARGINAL, `1` = REVIEW/FAIL, `2` = error.

---

## Configuration

PromptLens supports persistent configuration via `.promptlensrc.toml` (searched up from cwd) or `pyproject.toml`:

**.promptlensrc.toml:**
```toml
provider = "together"        # together | openai | anthropic
model = ""                   # empty = provider default
threshold = 0.15
saliency_threshold = 0.15
m_samples = 20
semantic = false
output_format = "terminal"   # terminal | json | sarif
verbose = false
```

**pyproject.toml:**
```toml
[tool.promptlens]
provider = "openai"
model = "gpt-4o-mini"
threshold = 0.10
output_format = "json"
```

**Precedence:** CLI flags > environment variables > `.promptlensrc.toml` > `pyproject.toml` > defaults.

**Environment variables:**
```bash
PROMPTLENS_PROVIDER=openai
PROMPTLENS_THRESHOLD=0.10
PROMPTLENS_FORMAT=json
PROMPTLENS_VERBOSE=true
PROMPTLENS_M_SAMPLES=50
```

---

## Multi-Provider Support

PromptLens supports three LLM providers with different capability levels:

| Provider | Generation | Embeddings | Semantic Mode | Notes |
|----------|:----------:|:----------:|:-------------:|-------|
| Together AI | ✓ | ✓ | ✓ | Default. Free $1 credit. Llama 3.3 70B + nomic-embed. |
| OpenAI | ✓ | ✓ | ✓ | GPT-4o-mini + text-embedding-3-small. |
| Anthropic | ✓ | — | — | Claude Sonnet. Generation only; `--semantic` not supported. |

Configure via `.promptlensrc.toml`, `pyproject.toml`, or environment variables:

```bash
# Use OpenAI
export OPENAI_API_KEY=sk-...
promptlens audit --repo .   # with provider = "openai" in config

# Or override per-run via env
PROMPTLENS_PROVIDER=openai promptlens check --file prompt.txt
```

All providers include automatic retry with exponential backoff, rate-limit awareness (respects `Retry-After` headers), and per-request timeouts.

---

## CI Integration

### GitHub Actions

Use the built-in composite action:

```yaml
# .github/workflows/promptlens.yml
name: Prompt Audit
on: [pull_request]

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./agent
        with:
          together-api-key: ${{ secrets.TOGETHER_API_KEY }}
          format: sarif
          threshold: '0.15'
```

This installs PromptLens, runs `promptlens audit --repo . --ci`, and uploads SARIF results to GitHub's code scanning dashboard.

### Manual CI setup

```bash
pip install promptlens-sdk promptlens-agent
promptlens audit --repo . --format sarif --ci > results.sarif
# Upload results.sarif to your code scanning tool
```

### Exit codes for CI

| Code | Meaning |
|------|---------|
| `0` | No issues found / compression passed |
| `1` | Issues found (redundancy > 20%) or compression failed validation |
| `2` | Runtime error |

### Structured output formats

**JSON** (`--format json`): Machine-readable output with per-phrase scores, dispositions, character offsets, and explainability rationale.

**SARIF** (`--format sarif`): [Static Analysis Results Interchange Format](https://sarifweb.azurewebsites.net/) v2.1.0. Each low-saliency phrase is a finding with file location, severity level, and actionable message. Compatible with GitHub Code Scanning, Azure DevOps, and other SARIF consumers.

---

## Python SDK

For programmatic access:

```python
import asyncio
from promptlens import run_shapley, SimilarityMode

report = asyncio.run(run_shapley(
    prompt="You are a helpful assistant. Always respond concisely. Use bullet points.",
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

### Provider configuration

```python
from promptlens import TogetherProvider, OpenAIProvider, configure_provider

# Switch to OpenAI
configure_provider(OpenAIProvider(api_key="sk-..."))

# Or use separate providers for different capabilities
from promptlens import AnthropicProvider
configure_provider(
    AnthropicProvider(api_key="sk-ant-..."),           # generation
    embedding_provider=OpenAIProvider(api_key="sk-...") # embeddings
)
```

### Key types

```python
SaliencyReport:
    scores: list[SaliencyScore]    # .score (0–1), .raw_shapley, .disposition
    phrases: list[Phrase]          # .text, .char_start, .char_end, .region_type
    token_count: int
    redundancy_fraction: float
    compression_candidate_tokens: int
    confidence: float

Phrase:
    text: str                      # display text
    index: int
    char_start: int                # character offset in original prompt
    char_end: int
    source_text: str               # exact original slice
    region_type: RegionType        # PLAIN, BULLETS, JSON, XML_TAGGED, CODE_BLOCK
    atomic: bool                   # if True, never removed (e.g. code blocks)
```

---

## How Attribution Works

Standard perturbation methods (leave-one-out, ablation) test phrases in isolation and miss interactions. A phrase can look useless alone but be essential when combined with others.

[Shapley values](https://en.wikipedia.org/wiki/Shapley_value) fix this. The Shapley value for a phrase is its **average marginal contribution across all possible coalitions** — the only attribution method satisfying all four fairness axioms (efficiency, symmetry, dummy, additivity).

**Computation:**
- **N ≤ 4 phrases:** Exact computation (all 2^N subsets).
- **N > 4 phrases:** Monte Carlo sampling with M random coalition walks, concurrency-capped at 5.
- **Caching:** Coalition outputs are cached — concurrent walks hitting the same subset share a single API call.
- **Determinism:** All generation uses `temperature: 0.0` for stable divergence measurement.

**Similarity modes:**
- **Standard (default):** Character trigram cosine distance. No extra API calls. Fast.
- **Semantic (`--semantic`):** Embedding cosine distance. Captures meaning-level change — rewording the same idea doesn't register as divergence.

---

## Web Tool

Paste a prompt in the browser and get it back colour-coded by impact. No install required.

**URL:** https://rohityalavarthy.github.io/PromptLens

**Run locally:**
```bash
cd web && python3 -m http.server 8080
# open http://localhost:8080
```

**Features:**
- Heatmap and table views with sortable scores
- Threshold slider to focus on high-impact content
- In-browser compression with structural cleanup
- Copy as markdown table or export full JSON
- Shareable URL (encodes analysis state in fragment)
- Run history (last 8 analyses in localStorage)
- Supports Groq, Together AI, OpenAI, and Anthropic (keys stored in localStorage only)

---

## Development

```bash
# Install in development mode
pip install -e sdk/python
pip install -e "agent[dev]"

# Run tests (167 total, all offline — no API keys needed)
cd sdk/python && pytest tests/ -v    # 41 tests
cd agent && pytest tests/ -v         # 126 tests
```

### Project structure

```
PromptLens/
├── sdk/python/promptlens/       # Core SDK
│   ├── shapley.py               # Shapley value computation
│   ├── segmenter.py             # Prompt segmentation with span tracking
│   ├── similarity.py            # Trigram + embedding similarity
│   ├── generator.py             # LLM generation (delegates to providers)
│   ├── provider.py              # Provider abstraction (Together/OpenAI/Anthropic)
│   ├── resilience.py            # Retry, timeout, backoff
│   └── types.py                 # Phrase, Region, SaliencyScore, etc.
├── agent/promptlens_agent/      # CLI Agent
│   ├── cli.py                   # Click CLI (check/audit/compress)
│   ├── discovery.py             # Python AST-based prompt discovery
│   ├── discovery_ts.py          # TypeScript/JavaScript discovery
│   ├── discovery_config.py      # YAML/template discovery
│   ├── compressor.py            # LLM compression with ID-based parsing
│   ├── validator.py             # Divergence validation loop
│   ├── formatters.py            # JSON/SARIF output formatters
│   ├── config.py                # Configuration file loading
│   ├── cost_estimator.py        # API cost estimation
│   └── reporter.py              # Terminal output formatting
├── agent/action.yml             # GitHub Actions composite action
└── web/                         # Browser-based tool
```

---

## License

MIT
