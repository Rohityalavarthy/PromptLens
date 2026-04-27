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

1. **Provider config** — `PROVIDERS` object holds Groq and Together AI endpoints, model IDs, `localStorage` keys, and key-format validation hints. Both providers use the OpenAI-compatible `/v1/chat/completions` format, so `callLLM()` is identical for both. Together AI also provides embeddings via `/v1/embeddings` using `nomic-ai/nomic-embed-text-v1.5`.
2. **State** — three mutable globals: `shapleyM` (sample count, default 20), `selectedProvider`, `analysisTarget` (`'user'|'system'`), `selectedSimilarity`. API keys are stored in `localStorage`, never in JS state.
3. **`runAnalysis()`** — the main orchestrator. Reads `#prompt` and `#systemprompt`, decides which is the analysis subject via `analysisTarget`, calls `segmentPrompt()`, gets a baseline response, calls `computeShapley()`, then `normalise()` → `renderSaliency()` → `renderStats()`.
4. **Segmentation** — classify-then-segment pipeline. `detectRegions()` scans the prompt line-by-line and classifies it into typed regions (`plain`, `code_block`, `json`, `bullets`, `xml_tagged`, `markdown`). Each region is passed to a dedicated segmenter returning `Phrase[]` objects `{ text, atomic, tagName?, content? }`.
5. **Shapley Attribution** — `computeShapley()` is the saliency engine. For N ≤ 4 phrases it computes exact Shapley values (all N! permutations). For N > 4 it uses Monte Carlo sampling with `shapleyM` random walks, concurrency-capped at 5 via `withConcurrency()`. Each walk calls `sampleShapleyWalk()` which incrementally builds a coalition and records each phrase's marginal contribution. A coalition cache (`makeCoalitionRunner()`) deduplicates repeated subset API calls across walks.
6. **Similarity modes** — controlled by `selectedSimilarity`. Standard: `1 - cosineSim(a, b)` (character trigram). Semantic: `1 - vecCosineSim(embed(a), embed(b))` (embedding cosine via Together AI — no LLM judge in Shapley). Semantic mode requires a Together AI key; the similarity selector is hidden until one is saved.
7. **`scoreToBackground()`** — maps a normalised [0,1] score to an RGBA colour via 5-stop gradient interpolation (dark-blue → cyan → amber → orange → red).

### Python SDK (`sdk/python/promptlens/`)

- `temperature: 0.0` on all LLM calls — determinism is essential for stable divergence measurement.
- `#prompt` is always the user message and `#systemprompt` is always the system message in the DOM regardless of which is being analyzed. The `buildCall()` closure inside `runAnalysis()` routes coalition text to the correct API role.
- Coalition cache stores Promises (not resolved values) — concurrent walks that hit the same coalition share a single in-flight API call rather than making duplicates.
- Absent XML-tagged phrases in a coalition are replaced with `<tag>[...]</tag>` to preserve document structure; absent plain phrases are omitted entirely.
- Textarea visual hierarchy swaps on target toggle: the analyzed textarea is tall/prominent (`.textarea--primary`), the context textarea is short/dimmed (`.textarea--secondary`). `selectTarget()` swaps these classes with a CSS transition.
- CSS tooltips on the similarity buttons use `[data-tooltip]::after` — no JS required.

### CLI Agent (`agent/promptlens_agent/`)

- **`discovery.py`** — `PythonPromptVisitor` extends `ast.NodeVisitor`. Tracks string assignments for variable resolution. Detects `messages=[{"role": "system", ...}]` patterns across OpenAI, Anthropic, LangChain, Bedrock signatures. Resolves `file` origin by reading the referenced path. Skips `.venv`, `venv`, `node_modules`, `__pycache__`, `.git`.
- **`compressor.py`** — sends KEEP/COMPRESS-labelled phrases to Qwen rewriter. Parses `[KEEP]`, `[REMOVE]`, `[MERGE]`, `[REWRITE]` response lines back to diff entries. Falls back to original on unparseable lines.
- **`validator.py`** — runs original and compressed prompts against all test inputs, computes divergence per input. On failure: reinstates the first removable phrase in the failing set, rebuilds the prompt, retries. MAX_RETRIES=3.
- **`tools.py`** — `@tool`-decorated wrappers for LangGraph orchestration: `scan_codebase`, `analyze_prompt`, `compress_and_validate`. These are not used by the CLI directly — they're the hook point for a future smart-mode agent graph.
- **`cli.py`** — `_load_test_inputs` with no file returns 3 hardcoded generic inputs (no LLM synthetic generation in v1).
- Agent imports SDK as an installed package: `from promptlens import ...` (not relative imports).
