# PromptLens

A browser-based saliency debugger for prompt engineers and developers building LLM applications. Paste a prompt, run the analysis, and get your prompt back colour-coded by how much each phrase actually influences the model output.

Built this because I kept iterating on prompts with no real signal on what was doing work and what wasn't. Most debugging is just vibes - this makes it measurable.

**Run it directly at:** https://rohityalavarthy.github.io/PromptLens

---

## What it does

PromptLens runs a Shapley attribution analysis on your prompt. It splits the prompt into phrases, then measures each phrase's contribution by testing it in combination with every possible subset of the other phrases — not just in isolation. The result is your original prompt rendered as colour-coded text - blue for low impact, red for high impact - with exact percentage scores on hover.

You can analyze either the **user prompt** or the **system prompt**, which makes it useful both for end-user prompt engineering and for developers tuning system prompts in production applications.

<img width="1414" height="810" alt="Screenshot 2026-04-26 at 8 51 39 PM" src="https://github.com/user-attachments/assets/ee13a2b0-1ee8-450b-8196-566ac70064b7" />


---

## How it works

### 1. Structure-aware segmentation

The prompt is classified into structural regions first — plain prose, JSON, XML tags, bullet lists, fenced code blocks, markdown sections — then each region is segmented by its own rules. Plain text splits at sentence and clause boundaries. JSON splits per top-level key. XML splits per outermost tag. Bullets split per item. Code blocks and markdown headers are kept atomic.

This gives phrase-level granularity that respects the structure of the prompt rather than blindly splitting on punctuation.

### 2. Baseline generation

The full unmodified prompt is sent to the model once to get a reference output. All subsequent comparisons are made against this baseline.

### 3. Shapley attribution

The three classic perturbation methods (Perturbation, Leave-One-Out, Paraphrase) all share the same flaw: they test each phrase in isolation, which misses interactions between phrases. A phrase can look low-impact on its own but be essential in combination with others.

Shapley values solve this correctly. The Shapley value for a phrase is its **average marginal contribution across all possible coalitions of other phrases** — the only attribution method that satisfies all fairness axioms when features interact.

Since exact computation requires 2^N model calls, PromptLens uses **Monte Carlo sampling**: M random coalition walks are run in parallel (concurrency cap 5), each walk recording the marginal contribution of each phrase as it's added to a growing coalition. M=20 walks gives a statistically stable estimate in roughly 20×N model calls.

For short prompts (N ≤ 4) exact Shapley is computed instead — all N! permutations share a coalition cache, so actual API calls stay well under 2^N.

**API call budget:**

| Prompt size | Walks | Est. model calls |
|---|---|---|
| N ≤ 4 | All N! permutations (exact) | ≤ 16 (cached) |
| N = 10, M = 20 | 20 | ~20–40 (with cache hits) |
| N = 10, M = 50 | 50 | ~50–80 (with cache hits) |

The sample count is configurable via the **Advanced** panel: Fast (10), Balanced (20), Precise (50).

### 4. Divergence measurement

Two modes are available:

**Standard** — character trigram cosine similarity. Overlapping 3-character substrings are extracted from both outputs, frequency vectors are built, and cosine distance is computed. Language-agnostic, no extra API calls, sensitive to surface-level rewording.

**Semantic** — embedding cosine distance via `nomic-ai/nomic-embed-text-v1.5`. Each pair of adjacent-step outputs in a Shapley walk is embedded and compared in vector space. Captures meaning-level change rather than surface change — a phrase that causes the model to say the same thing differently scores low rather than high. Requires a Together AI key.

### 5. Normalisation and rendering

Raw Shapley scores are min-max normalised across all phrases so the full colour range is always used. Each phrase is rendered as an inline span with a background interpolated across a 5-stop gradient (dark blue → cyan → amber → orange → red). Hovering any phrase shows its exact impact percentage.

---

## Analyzing system prompts

The toggle at the top of the input panel switches between analyzing the user prompt and the system prompt. Whichever is selected gets the saliency treatment; the other is held constant as context throughout all coalition walks.

This is the main reason I built support for system prompt analysis rather than keeping it user-prompt-only - if you're a developer shipping an LLM feature, your system prompt is where most of the complexity lives and it's exactly what you want to be able to debug.

When analyzing the system prompt, make sure to fill in a representative user message in the lower field. The quality of the saliency signal depends on having a realistic user turn for the model to respond to.

---

## Getting started

No build step, no dependencies, no backend. It's three files.

### Run locally

```bash
git clone https://github.com/Rohityalavarthy/PromptLens.git
cd PromptLens
python3 -m http.server 8080
# open http://localhost:8080
```

You'll see a 404 for `/favicon.ico` in the server logs - that's just the browser looking for a tab icon, it's harmless.

### Deploy to GitHub Pages

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/Rohityalavarthy/PromptLens.git
git push -u origin main
```

Then go to **Settings → Pages → Source** and set it to `main` branch, root folder. GitHub will give you a live URL within about a minute. No config needed - it's a static site.

---

## API keys

PromptLens supports two providers. Keys are stored in `localStorage` and never leave your browser - every request goes directly from your browser to the provider's API.

### Groq - recommended for Standard mode

Runs `llama-3.3-70b-versatile`. Fastest inference of any free API available, very generous rate limits.

1. Sign up at [console.groq.com](https://console.groq.com/keys) - no credit card required
2. Create an API key (starts with `gsk_`)
3. Paste it into the key modal in PromptLens

Free tier: ~30 requests/min. A Balanced (M=20) Shapley run on a 10-phrase prompt uses roughly 20–40 calls with cache hits.

### Together AI - required for Semantic mode

Runs `meta-llama/Llama-3.3-70B-Instruct-Turbo` for generation and `nomic-ai/nomic-embed-text-v1.5` for embeddings. A Together AI key is required to unlock Semantic similarity mode.

1. Sign up at [api.together.ai](https://api.together.ai/settings/api-keys) - comes with $1 free credit, no credit card
2. Create an API key and paste it in

Once a Together AI key is saved, the Similarity selector appears and defaults to Semantic mode.

---

## File structure

```
promptlens/
├── index.html    # markup and layout
├── style.css     # all styles, CSS custom properties for theming
├── app.js        # segmentation, Shapley attribution, API calls, rendering
└── README.md
```

`app.js` is organized into clearly commented sections: provider config → state → key management → method/target selection → LLM call → segmentation → similarity → Shapley attribution → normalisation → colour mapping → rendering → main analysis runner.

---

## Future Roadmap

- [ ] Export saliency map as image
- [ ] Side-by-side diff of two prompt variants
- [ ] Batch mode - run the same analysis across multiple test inputs and aggregate scores
- [ ] OpenAI and Anthropic key support

---

## License

MIT
