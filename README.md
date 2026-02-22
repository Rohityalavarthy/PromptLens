# PromptLens

A browser-based saliency debugger for prompt engineers and developers building LLM applications. Paste a prompt, run the analysis, and get your prompt back colour-coded by how much each phrase actually influences the model output.

Built this because I kept iterating on prompts with no real signal on what was doing work and what wasn't. Most debugging is just vibes - this makes it measurable.

---

## What it does

PromptLens runs a perturbation-based saliency analysis on your prompt. It splits the prompt into phrases, systematically perturbs each one, re-runs the prompt, and measures how much the output changes. The more the output diverges when a phrase is modified, the higher that phrase's saliency score. The result is your original prompt rendered as colour-coded text - blue for low impact, red for high impact - with exact percentage scores on hover.

You can analyze either the **user prompt** or the **system prompt**, which makes it useful both for end-user prompt engineering and for developers tuning system prompts in production applications.

![PromptLens screenshot](assets/view.png)

---

## How it works

### 1. Phrase tokenization

The prompt is split at sentence boundaries (`.`, `!`, `?`, newlines), then long sentences are further split at clause boundaries (`,`, `;`) until each chunk is roughly 35–60 characters. This gives phrase-level granularity without the noise of token-level coloring.

### 2. Baseline generation

The full unmodified prompt is sent to the model once to get a reference output. All subsequent comparisons are made against this baseline.

### 3. Saliency computation

Three methods are available:

| Method | Mechanism | API calls | Best for |
|---|---|---|---|
| **Perturbation** | Replaces each phrase with `[...]` | N+1 | General use, fastest |
| **Leave-One-Out** | Removes each phrase entirely | N+1 | Short, dense prompts |
| **Paraphrase** | Asks the model to rewrite each phrase as something maximally vague, then re-runs | 2N+1 | Most semantically accurate signal |

### 4. Divergence measurement

Output similarity is computed using **character trigram cosine similarity** - overlapping 3-character substrings are extracted from both outputs, frequency vectors are built, and cosine distance is computed. Saliency score = `1 - similarity`. This is language-agnostic, requires no NLP library, and is robust to surface-level paraphrasing in the model output.

Chose this technique on this basis of: **Smolka, A., Wang, H.-M., Chang, J. S. et al. (2022). Is Character Trigram Overlapping Ratio Still the Best for Aligning Paraphrases?**

But due to the apparent limitations of this technique, I will be looking towards moving towards a better saliency measurement formula in the next iteration.

### 5. Normalisation and rendering

Raw scores are min-max normalised across all phrases so the full colour range is always used. Each phrase is rendered as an inline span with a background interpolated across a 5-stop gradient (dark blue → cyan → amber → orange → red). Hovering any phrase shows its exact impact percentage.

---

## Analyzing system prompts

The toggle at the top of the input panel switches between analyzing the user prompt and the system prompt. Whichever is selected gets the saliency treatment; the other is held constant as context throughout all perturbation passes.

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

### Groq - recommended

Runs `llama-3.3-70b-versatile`. Fastest inference of any free API available, very generous rate limits.

1. Sign up at [console.groq.com](https://console.groq.com/keys) - no credit card required
2. Create an API key (starts with `gsk_`)
3. Paste it into the key modal in PromptLens

Free tier: ~30 requests/min, ~14,400/day. A standard perturbation analysis on a 10-phrase prompt uses 11 calls, so you have a lot of headroom.

### Together AI - alternative

Runs `meta-llama/Llama-3.3-70B-Instruct-Turbo`. Same model family as Groq, useful fallback if you're hitting rate limits.

1. Sign up at [api.together.ai](https://api.together.ai/settings/api-keys) - comes with $1 free credit, no credit card
2. Create an API key and paste it in

Both providers use the OpenAI-compatible `/v1/chat/completions` format so the same request logic works for both.

---

## File structure

```
promptlens/
├── index.html    # markup and layout
├── style.css     # all styles, CSS custom properties for theming
├── app.js        # tokenization, saliency methods, API calls, rendering
└── README.md
```

`app.js` is organized into clearly commented sections: provider config → state → key management → method/target selection → LLM call → tokenization → similarity → saliency methods → normalisation → colour mapping → rendering → main analysis runner.

---

## Future Roadmap

- [ ] Export saliency map as image
- [ ] Side-by-side diff of two prompt variants
- [ ] Batch mode - run the same analysis across multiple test inputs and aggregate scores
- [ ] OpenAI and Anthropic key support
- [ ] Implement a better token level saliency measure which is then normalized to phrases

---

## License

MIT
