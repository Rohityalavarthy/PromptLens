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

- **`index.html`** — layout and markup only; no inline scripts
- **`style.css`** — all styles using CSS custom properties defined in `:root`; dark theme with a CRT scanline overlay via `body::before`
- **`app.js`** — all logic; organized into clearly labelled sections (see below)

### app.js data flow

1. **Provider config** — `PROVIDERS` object holds Groq and Together AI endpoints, model IDs, `localStorage` keys, and key-format validation hints. Both providers use the OpenAI-compatible `/v1/chat/completions` format, so `callLLM()` is identical for both.
2. **State** — three mutable globals: `selectedMethod`, `selectedProvider`, `analysisTarget` (`'user'|'system'`). API keys are stored in `localStorage`, never in JS state.
3. **`runAnalysis()`** — the main orchestrator. It reads `#prompt` (user message) and `#systemprompt` (system message), decides which is the analysis subject via `analysisTarget`, calls `tokenizePhrases()`, gets a baseline response, dispatches to the chosen saliency method, then calls `normalise()` → `renderSaliency()` → `renderStats()`.
4. **`tokenizePhrases()`** — splits at sentence-ending punctuation first, then sub-splits long sentences at clause boundaries (`,`/`;`), accumulating until chunks reach ~35 characters.
5. **Saliency methods** — `saliencyPerturbation`, `saliencyOmission`, `saliencyParaphrase` each accept `(phrases, baseline, callPerturbed, onTick)`. They return raw divergence scores (1 − cosine similarity).
6. **`cosineSim()`** — character trigram cosine similarity; no external NLP dependency.
7. **`scoreToBackground()`** — maps a normalised [0,1] score to an RGBA colour via 5-stop gradient interpolation (dark-blue → cyan → amber → orange → red).

### Key design decisions

- `temperature: 0.0` on all LLM calls — determinism is essential for stable divergence measurement.
- `#prompt` is always the user message and `#systemprompt` is always the system message in the DOM regardless of which is being analyzed. The `buildCall()` closure inside `runAnalysis()` routes the perturbed text to the correct API role.
- Paraphrase method costs 2× API calls per phrase (one to neutralise the phrase, one to re-run with the neutralised version).
- `app.js?v=5` cache-busting query string on the script tag in `index.html` — increment this when deploying changes.

## Roadmap (from README)

- Export saliency map as image
- Side-by-side diff of two prompt variants
- Batch mode across multiple test inputs
- OpenAI and Anthropic provider support
- Token-level saliency measure normalised to phrases (to replace character trigram cosine similarity)
