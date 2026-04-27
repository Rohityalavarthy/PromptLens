# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

PromptLens is a Shapley-value attribution tool for LLM prompts. The repo is a monorepo with three components:

- **`web/`** — browser-based saliency debugger (static, no build step)
- **`sdk/python/`** — Python SDK (`promptlens-sdk` package): segmentation, Shapley engine, similarity
- **`agent/`** — CLI tool (`promptlens-agent` package): codebase prompt discovery, compression, validation

**Live site:** https://rohityalavarthy.github.io/PromptLens

---

## Running things

### Web tool

```bash
cd web && python3 -m http.server 8080
# open http://localhost:8080
```

The `/favicon.ico` 404 in server logs is harmless. GitHub Pages is deployed via `.github/workflows/pages.yml` from the `web/` directory (not the repo root).

### SDK + agent (Python 3.11+)

```bash
pip install -e sdk/python
pip install -e agent
export TOGETHER_API_KEY=your_key_here
```

### Tests

```bash
# SDK — 19 tests, all offline (no API key needed)
cd sdk/python && pytest tests/ -v

# Agent — 6 tests, all offline
cd agent && pytest tests/ -v
```

### CLI

```bash
promptlens check --file path/to/prompt.txt          # fast, M=3
promptlens audit --repo .                           # full repo scan
promptlens compress --file path/to/prompt.txt       # analyse → rewrite → validate
```

---

## Repo structure

```
web/
  index.html      layout only, no inline scripts
  app.js          all web logic (cache-bust with ?v=N on changes)
  style.css       all styles, CSS custom properties in :root (cache-bust with ?v=N)
  assets/

sdk/python/
  promptlens/
    types.py      dataclasses: Phrase, Region, SaliencyScore, SaliencyReport, CompressionResult
    segmenter.py  detect_regions() + per-type segmenters → segment_prompt()
    generator.py  generate(), get_embedding(), judge_divergence() — all Together AI
    similarity.py trigram_similarity(), semantic_divergence(), compute_divergences()
    shapley.py    run_shapley() — main entry point; CoalitionCache, sample_coalition_walk()
    __init__.py   public surface: run_shapley, segment_prompt, types
  tests/
    test_segmenter.py
    test_shapley.py    (pure logic only — no API calls)
    test_similarity.py (pure logic only — no API calls)
  pyproject.toml  name=promptlens-sdk; hatch wheel target: packages=["promptlens"]

agent/
  promptlens_agent/
    discovery.py    PythonPromptVisitor (AST), discover_prompts()
    compressor.py   compress_prompt() — labelled LLM rewrite
    validator.py    validate_compression() — divergence loop, MAX_RETRIES=3
    reporter.py     ANSI terminal output, no external deps
    tools.py        @tool-decorated LangGraph wrappers (scan_codebase, analyze_prompt, compress_and_validate)
    cli.py          Click entry point: check / audit / compress commands
  tests/
    test_discovery.py
  pyproject.toml  name=promptlens-agent; entry point: promptlens = promptlens_agent.cli:main
```

---

## Architecture

### Web tool (`web/app.js`)

Data flow: `runAnalysis()` → `segmentPrompt()` → baseline call → `computeShapley()` → `normalise()` → `renderSaliency()` → `renderStats()`

Key decisions:
- `#systemprompt` is always the analysis subject (top, `.textarea--primary`). `#prompt` is always the held-constant test input (bottom, `.textarea--secondary`). No toggle — this is fixed.
- `buildCall(analyzedText)` always returns `{ userMsg: contextText, systemMsg: analyzedText }`.
- Coalition cache stores Promises (not resolved values) — concurrent walks sharing a coalition share one in-flight API call.
- Absent XML phrases in a coalition are replaced with `<tag>[...]</tag>`; absent plain phrases are omitted.
- CSS tooltips on similarity buttons use `[data-tooltip]::after` — no JS needed.
- Both `style.css` and `app.js` are cache-busted via `?v=N` query strings in `index.html`. Increment N when deploying changes.

### Python SDK (`sdk/python/promptlens/`)

- **`generator.py`** — all API calls go to Together AI. Models: `meta-llama/Llama-3.3-70B-Instruct-Turbo` (generation), `togethercomputer/m2-bert-80M-8k-retrieval` (embeddings), `Qwen/Qwen2.5-72B-Instruct-Turbo` (judge — different family for independence). `get_api_key()` reads `TOGETHER_API_KEY` env var.
- **`segmenter.py`** — `detect_regions()` classifies lines into typed regions; per-type segmenters return `Phrase` objects. Region types: `PLAIN`, `CODE_BLOCK`, `JSON`, `BULLETS`, `XML_TAGGED`, `MARKDOWN`.
- **`shapley.py`** — `run_shapley()` orchestrates everything. For N ≤ 4 phrases: exact (all permutations). For N > 4: Monte Carlo with `m_samples` walks, semaphore cap 5. `CoalitionCache` keys on sorted index set. `sample_coalition_walk()` calls `generate(user_input, system_prompt=prompt)` — system prompt in the system role, user input in the user role.
- **`similarity.py`** — STANDARD: trigram cosine distance. SEMANTIC: embed → cosine distance → z-score filter → Qwen judge on outliers.
- `temperature=0.0` on all generation calls.

### CLI Agent (`agent/promptlens_agent/`)

- **`discovery.py`** — `PythonPromptVisitor` extends `ast.NodeVisitor`. Tracks string assignments for variable resolution. Detects `messages=[{"role": "system", ...}]` patterns across OpenAI, Anthropic, LangChain, Bedrock signatures. Resolves `file` origin by reading the referenced path. Skips `.venv`, `venv`, `node_modules`, `__pycache__`, `.git`.
- **`compressor.py`** — sends KEEP/COMPRESS-labelled phrases to Qwen rewriter. Parses `[KEEP]`, `[REMOVE]`, `[MERGE]`, `[REWRITE]` response lines back to diff entries. Falls back to original on unparseable lines.
- **`validator.py`** — runs original and compressed prompts against all test inputs, computes divergence per input. On failure: reinstates the first removable phrase in the failing set, rebuilds the prompt, retries. MAX_RETRIES=3.
- **`tools.py`** — `@tool`-decorated wrappers for LangGraph orchestration: `scan_codebase`, `analyze_prompt`, `compress_and_validate`. These are not used by the CLI directly — they're the hook point for a future smart-mode agent graph.
- **`cli.py`** — `_load_test_inputs` with no file returns 3 hardcoded generic inputs (no LLM synthetic generation in v1).
- Agent imports SDK as an installed package: `from promptlens import ...` (not relative imports).
