# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

PromptLens is a browser-based prompt saliency debugger. It has no build step, no dependencies, and no backend — just three files served statically.

**Live site:** https://rohityalavarthy.github.io/PromptLens

## Running locally

```bash
python3 -m http.server 8080
# open http://localhost:8080
```

The `/favicon.ico` 404 in server logs is harmless.

## Architecture

Everything lives in three files:

- **`index.html`** — layout and markup only; no inline scripts. Cache-bust both `style.css?v=N` and `app.js?v=N` query strings when deploying CSS or JS changes respectively.
- **`style.css`** — all styles using CSS custom properties defined in `:root`; dark theme with a CRT scanline overlay via `body::before`
- **`app.js`** — all logic; organized into clearly labelled sections (see below)

### app.js data flow

1. **Provider config** — `PROVIDERS` object holds Groq and Together AI endpoints, model IDs, `localStorage` keys, and key-format validation hints. Both providers use the OpenAI-compatible `/v1/chat/completions` format, so `callLLM()` is identical for both. Together AI also provides embeddings via `/v1/embeddings` using `nomic-ai/nomic-embed-text-v1.5`.
2. **State** — three mutable globals: `shapleyM` (sample count, default 20), `selectedProvider`, `analysisTarget` (`'user'|'system'`), `selectedSimilarity`. API keys are stored in `localStorage`, never in JS state.
3. **`runAnalysis()`** — the main orchestrator. Reads `#prompt` and `#systemprompt`, decides which is the analysis subject via `analysisTarget`, calls `segmentPrompt()`, gets a baseline response, calls `computeShapley()`, then `normalise()` → `renderSaliency()` → `renderStats()`.
4. **Segmentation** — classify-then-segment pipeline. `detectRegions()` scans the prompt line-by-line and classifies it into typed regions (`plain`, `code_block`, `json`, `bullets`, `xml_tagged`, `markdown`). Each region is passed to a dedicated segmenter returning `Phrase[]` objects `{ text, atomic, tagName?, content? }`.
5. **Shapley Attribution** — `computeShapley()` is the saliency engine. For N ≤ 4 phrases it computes exact Shapley values (all N! permutations). For N > 4 it uses Monte Carlo sampling with `shapleyM` random walks, concurrency-capped at 5 via `withConcurrency()`. Each walk calls `sampleShapleyWalk()` which incrementally builds a coalition and records each phrase's marginal contribution. A coalition cache (`makeCoalitionRunner()`) deduplicates repeated subset API calls across walks.
6. **Similarity modes** — controlled by `selectedSimilarity`. Standard: `1 - cosineSim(a, b)` (character trigram). Semantic: `1 - vecCosineSim(embed(a), embed(b))` (embedding cosine via Together AI — no LLM judge in Shapley). Semantic mode requires a Together AI key; the similarity selector is hidden until one is saved.
7. **`scoreToBackground()`** — maps a normalised [0,1] score to an RGBA colour via 5-stop gradient interpolation (dark-blue → cyan → amber → orange → red).

### Key design decisions

- `temperature: 0.0` on all LLM calls — determinism is essential for stable divergence measurement.
- `#prompt` is always the user message and `#systemprompt` is always the system message in the DOM regardless of which is being analyzed. The `buildCall()` closure inside `runAnalysis()` routes coalition text to the correct API role.
- Coalition cache stores Promises (not resolved values) — concurrent walks that hit the same coalition share a single in-flight API call rather than making duplicates.
- Absent XML-tagged phrases in a coalition are replaced with `<tag>[...]</tag>` to preserve document structure; absent plain phrases are omitted entirely.
- Textarea visual hierarchy swaps on target toggle: the analyzed textarea is tall/prominent (`.textarea--primary`), the context textarea is short/dimmed (`.textarea--secondary`). `selectTarget()` swaps these classes with a CSS transition.
- CSS tooltips on the similarity buttons use `[data-tooltip]::after` — no JS required.

## Roadmap (from README)

- Export saliency map as image
- Side-by-side diff of two prompt variants
- Batch mode across multiple test inputs
- OpenAI and Anthropic provider support
