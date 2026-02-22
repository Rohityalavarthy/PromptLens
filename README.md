# PromptLens 🔬

**A browser-based prompt saliency debugger for prompt engineers.**

Paste any prompt, click Analyze, and see your prompt colour-coded by how much each phrase contributes to the model's output. Red = high impact, blue = low impact. Useful for identifying which parts of your prompt are actually doing work — and which are redundant.

![PromptLens screenshot](https://via.placeholder.com/900x500/0a0a0b/00e5ff?text=PromptLens)

---

## How it works

1. **Phrase tokenization** — your prompt is split into sentences and sub-clauses.
2. **Baseline generation** — the full prompt is sent to Llama 3.3 70B (via Groq) to get a reference output.
3. **Saliency computation** — one of three methods perturbs each phrase one at a time and measures how much the output diverges from the baseline.
4. **Score normalisation** — raw divergence scores are min-max normalised so the full colour range is always used.
5. **Rendering** — each phrase is wrapped in a colour-coded span. Hover any phrase to see its exact impact percentage.

### Saliency methods

| Method | How it works | Best for |
|---|---|---|
| **Perturbation** | Replaces phrase with `[...]` | Fast, general purpose |
| **Leave-One-Out** | Removes phrase entirely | Short, dense prompts |
| **Paraphrase** | Asks model to rewrite phrase as vague filler, then re-runs | Most semantically accurate |

### Divergence measurement

Output divergence is computed via **character trigram cosine similarity**. After perturbing a phrase, the new output is compared to the baseline using overlapping 3-character substrings. Low similarity = high importance. This approach is language-agnostic, requires no NLP library, and is robust to minor paraphrasing.

---

## Getting started

### Option 1 — GitHub Pages (live demo)

Visit: **`https://<your-username>.github.io/promptlens`**

You'll need a free [Groq API key](#groq-api-key) to use it.

### Option 2 — Run locally

```bash
git clone https://github.com/<your-username>/promptlens.git
cd promptlens
# Open in browser — no build step needed
open index.html
# Or use any static server:
npx serve .
python3 -m http.server 8080
```

---

## API Keys

PromptLens runs entirely in your browser. You choose which provider to use — your key is saved to `localStorage` and is only ever sent to the provider's servers directly from your browser.

### Groq (recommended)

Runs **Llama 3.3 70B Versatile**. Fastest inference, most generous free tier.

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up free — no credit card required
3. Create a key (starts with `gsk_`) and paste it into PromptLens

Free tier limits: ~30 requests/min, ~14,400 req/day.

### Together AI

Runs **Llama 3.3 70B Instruct Turbo**. Good alternative if you hit Groq rate limits.

1. Go to [api.together.ai/settings/api-keys](https://api.together.ai/settings/api-keys)
2. Sign up free — $1 credit included, no credit card required
3. Create a key and paste it into PromptLens

**A single PromptLens analysis uses `N+1` API calls** where N is the number of phrases (typically 5–15). The paraphrase method uses `2N+1` calls. Both providers' free tiers handle this comfortably.

---

## Deploying to GitHub Pages

```bash
# 1. Create a repo on GitHub named "promptlens"
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/promptlens.git
git push -u origin main

# 2. Enable GitHub Pages in repo Settings → Pages → Source: main branch / root
# 3. Your app is live at https://<your-username>.github.io/promptlens
```

---

## File structure

```
promptlens/
├── index.html      # App shell and markup
├── style.css       # All styles
├── app.js          # Tokenization, saliency logic, Groq API calls, rendering
└── README.md
```

---

## Contributing

Issues and PRs welcome. Some ideas for improvement:
- [ ] Export saliency report as PDF/image
- [ ] Side-by-side comparison of two prompts
- [ ] Attention head visualisation (requires model API support)
- [ ] Batch analysis across multiple test cases
- [ ] Support for OpenAI / Anthropic keys as alternatives

---

## License

MIT
