/**
 * PromptLens — app.js
 *
 * Architecture:
 *  1. User selects a provider (Groq or Together AI) and supplies their API key,
 *     stored in localStorage — never sent anywhere except the chosen provider's servers.
 *  2. The prompt is tokenized into phrases (sentence → clause level).
 *  3. A baseline response is obtained from the selected model.
 *  4. One of three saliency methods computes an importance score per phrase
 *     by perturbing it and measuring output divergence via character-trigram cosine similarity.
 *  5. Scores are min-max normalised and rendered as inline colour spans.
 */

'use strict';

// ─── Provider config ──────────────────────────────────────────────────────────
//
// Both providers expose an OpenAI-compatible /chat/completions endpoint,
// so the same fetch logic works for both — only the URL, model, and key differ.

const PROVIDERS = {
  groq: {
    id:          'groq',
    label:       'Groq',
    model:       'llama-3.3-70b-versatile',
    endpoint:    'https://api.groq.com/openai/v1/chat/completions',
    storageKey:  'promptlens_groq_key',
    keyPrefix:   'gsk_',
    keyHint:     'gsk_••••••••••••••••••••••••',
    signupUrl:   'https://console.groq.com/keys',
    signupLabel: 'console.groq.com/keys',
    freeNote:    'Free tier · No credit card · ~30 req/min',
    rateLimitMsg:'Groq rate limit hit. Wait a moment then try again (free tier: ~30 req/min).',
  },
  together: {
    id:          'together',
    label:       'Together AI',
    model:       'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    endpoint:    'https://api.together.xyz/v1/chat/completions',
    storageKey:  'promptlens_together_key',
    keyPrefix:   null,   // Together keys have no fixed prefix
    keyHint:     '••••••••••••••••••••••••••••••••',
    signupUrl:   'https://api.together.ai/settings/api-keys',
    signupLabel: 'api.together.ai/settings/api-keys',
    freeNote:    'Free $1 credit on sign-up · No credit card required',
    rateLimitMsg:'Together AI rate limit hit. Wait a moment then try again.',
  },
};

// ─── Constants ────────────────────────────────────────────────────────────────

const METHOD_DESCRIPTIONS = {
  perturbation: 'Replaces each phrase with <code>[...]</code> and measures how much the model output changes. Fast and reliable for most prompts.',
  omission:     'Removes each phrase entirely and measures the output divergence. Purer signal but may create grammatical gaps.',
  paraphrase:   'Rewrites each phrase to be vague and uninformative, then measures divergence. Most semantically faithful — but 2× the API calls.',
};

// ─── State ────────────────────────────────────────────────────────────────────

let selectedMethod   = 'perturbation';
let selectedProvider = localStorage.getItem('promptlens_provider') || 'groq';
let analysisTarget   = 'user';   // 'user' | 'system'

// ─── Provider & key management ────────────────────────────────────────────────

function getProvider() {
  return PROVIDERS[selectedProvider];
}

function getKey(providerId) {
  const p = providerId ? PROVIDERS[providerId] : getProvider();
  return localStorage.getItem(p.storageKey) || '';
}

function updateKeyButton() {
  const key   = getKey();
  const p     = getProvider();
  const btn   = document.getElementById('keyBtn');
  const label = document.getElementById('keyBtnLabel');
  if (key) {
    btn.classList.add('has-key');
    label.textContent = `${p.label} ✓`;
  } else {
    btn.classList.remove('has-key');
    label.textContent = 'Add API Key';
  }
}

/** Populate modal fields to reflect whichever provider tab is active. */
function refreshModalForProvider(providerId) {
  const p = PROVIDERS[providerId];

  // Tab active states
  document.querySelectorAll('.provider-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.provider === providerId);
  });

  // Instructions
  document.getElementById('providerSignupUrl').href        = p.signupUrl;
  document.getElementById('providerSignupLabel').textContent = p.signupLabel;
  document.getElementById('providerFreeNote').textContent  = p.freeNote;
  document.getElementById('providerModel').textContent     = p.model;

  // Input
  const input = document.getElementById('keyInput');
  input.placeholder = p.keyHint;
  input.value       = getKey(providerId);

  document.getElementById('keyError').textContent = '';
}

function selectProviderTab(tab) {
  selectedProvider = tab.dataset.provider;
  localStorage.setItem('promptlens_provider', selectedProvider);
  refreshModalForProvider(selectedProvider);
}

function openKeyModal() {
  refreshModalForProvider(selectedProvider);
  document.getElementById('keyModal').classList.add('open');
  document.getElementById('modalBackdrop').classList.add('open');
  setTimeout(() => document.getElementById('keyInput').focus(), 80);
}

function closeKeyModal() {
  document.getElementById('keyModal').classList.remove('open');
  document.getElementById('modalBackdrop').classList.remove('open');
}

function saveKey() {
  const p   = PROVIDERS[selectedProvider];
  const val = document.getElementById('keyInput').value.trim();

  if (!val) {
    document.getElementById('keyError').textContent = 'Please paste your API key.';
    return;
  }
  if (p.keyPrefix && !val.startsWith(p.keyPrefix)) {
    document.getElementById('keyError').textContent =
      `${p.label} keys start with "${p.keyPrefix}" — double-check you've copied the full key.`;
    return;
  }

  localStorage.setItem(p.storageKey, val);
  updateKeyButton();
  closeKeyModal();
}

function clearKey() {
  const p = PROVIDERS[selectedProvider];
  localStorage.removeItem(p.storageKey);
  document.getElementById('keyInput').value = '';
  document.getElementById('keyError').textContent = '';
  updateKeyButton();
}

// ─── Method selector ──────────────────────────────────────────────────────────

function selectMethod(btn) {
  document.querySelectorAll('.method-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedMethod = btn.dataset.method;
  document.getElementById('methodDesc').innerHTML = METHOD_DESCRIPTIONS[selectedMethod];
}

// ─── Analysis target selector ─────────────────────────────────────────────────

/**
 * Switch which prompt (user or system) is the analysis subject.
 * The other becomes the fixed context, held constant during perturbation.
 */
function selectTarget(btn) {
  document.querySelectorAll('.target-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  analysisTarget = btn.dataset.target;

  const analyzingUser = analysisTarget === 'user';

  // Primary label (the one being analyzed)
  document.getElementById('primaryLabel').innerHTML =
    `${analyzingUser ? 'User Prompt' : 'System Prompt'} <span class="target-badge">Analyzed</span>`;

  // Secondary label (held constant)
  document.getElementById('secondaryLabel').innerHTML =
    `${analyzingUser ? 'System Prompt' : 'User Prompt'} <span class="optional" id="secondaryOptional">(held constant)</span>`;

  // Placeholder text for the primary textarea
  document.getElementById('prompt').placeholder = analyzingUser
    ? 'Enter the user prompt to analyze. Each phrase will be colour-coded by impact.\n\nExample:\nSuggest three dinner recipes that are quick to make, use chicken, and are suitable for a family of four.'
    : 'Enter the system prompt to analyze. Each phrase will be colour-coded by impact.\n\nExample:\nYou are an expert chef. Always respond in a friendly tone. Focus on healthy ingredients. Avoid processed foods. Keep recipes under 30 minutes.';

  // Placeholder for the secondary (context) textarea
  document.getElementById('systemprompt').placeholder = analyzingUser
    ? 'Optional system prompt — held constant during analysis...'
    : 'Enter a fixed user message that will stay constant during analysis.\n\nExample:\nSuggest three dinner recipes for a family of four.';
}



/**
 * Call the active provider's chat completions endpoint.
 * Both Groq and Together AI use the OpenAI-compatible format.
 *
 * @param {string} userMsg
 * @param {string} systemMsg
 * @param {number} maxTokens
 * @returns {Promise<string>}
 */
async function callLLM(userMsg, systemMsg = '', maxTokens = 500) {
  const p   = getProvider();
  const key = getKey();

  if (!key) {
    throw new Error(`No ${p.label} API key set. Click "Add API Key" to add your key.`);
  }

  const messages = [];
  if (systemMsg) messages.push({ role: 'system', content: systemMsg });
  messages.push({ role: 'user', content: userMsg });

  const res = await fetch(p.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${key}`,
    },
    body: JSON.stringify({
      model:       p.model,
      messages,
      max_tokens:  maxTokens,
      temperature: 0.0,   // deterministic — essential for stable divergence measurement
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = err?.error?.message || `HTTP ${res.status}`;
    if (res.status === 401) throw new Error(`Invalid ${p.label} API key. Check your key and try again.`);
    if (res.status === 429) throw new Error(p.rateLimitMsg);
    throw new Error(`${p.label} API error: ${msg}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? '';
}

// ─── Phrase tokenisation ──────────────────────────────────────────────────────

/**
 * Split a prompt into phrases at sentence and clause boundaries.
 * Preserves all whitespace so the reconstructed text is identical to the input.
 *
 * Strategy:
 *  1. Split at sentence-ending punctuation (. ! ? \n).
 *  2. For sentences > 60 chars, further split at commas / semicolons,
 *     accumulating until a chunk is at least 35 chars long.
 *
 * @param {string} text
 * @returns {string[]}
 */
function tokenizePhrases(text) {
  // Regex: capture everything up to and including a sentence terminator, OR the remainder
  const sentRe = /([^.!?\n]+[.!?\n]+)|([^.!?\n]+$)/g;
  const sentences = [];
  let m;
  while ((m = sentRe.exec(text)) !== null) {
    const s = m[0];
    if (!s.trim()) continue;

    if (s.length > 60) {
      // Sub-split on comma / semicolon
      const parts = s.split(/(?<=[,;])/);
      let acc = '';
      for (let i = 0; i < parts.length; i++) {
        acc += parts[i];
        const isLast = i === parts.length - 1;
        if (acc.trim().length >= 35 || isLast) {
          if (acc.trim()) sentences.push(acc);
          acc = '';
        }
      }
      if (acc.trim()) sentences.push(acc);
    } else {
      sentences.push(s);
    }
  }

  return sentences.length > 0 ? sentences : [text];
}

// ─── Similarity ───────────────────────────────────────────────────────────────

/**
 * Character n-gram frequency vector.
 * @param {string} text
 * @param {number} n
 * @returns {Object<string,number>}
 */
function ngramFreq(text, n = 3) {
  const v = {};
  const t = text.toLowerCase();
  for (let i = 0; i <= t.length - n; i++) {
    const g = t.slice(i, i + n);
    v[g] = (v[g] || 0) + 1;
  }
  return v;
}

/**
 * Cosine similarity between two strings using character trigrams.
 * Returns value in [0, 1].
 */
function cosineSim(a, b) {
  const va = ngramFreq(a);
  const vb = ngramFreq(b);
  let dot = 0, na = 0, nb = 0;
  const keys = new Set([...Object.keys(va), ...Object.keys(vb)]);
  for (const k of keys) {
    const ai = va[k] || 0;
    const bi = vb[k] || 0;
    dot += ai * bi;
    na  += ai * ai;
    nb  += bi * bi;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ─── Saliency methods ─────────────────────────────────────────────────────────

/**
 * Perturbation: replace phrase[i] with '[...]', re-run, measure divergence.
 */
async function saliencyPerturbation(phrases, baseline, callPerturbed, onTick) {
  const scores = [];
  for (let i = 0; i < phrases.length; i++) {
    const perturbed = phrases.map((p, j) => (j === i ? '[...]' : p)).join('');
    try {
      const out = await callPerturbed(perturbed);
      scores.push(1 - cosineSim(baseline, out));
    } catch {
      scores.push(0);
    }
    onTick(i + 1, phrases.length);
  }
  return scores;
}

/**
 * Leave-one-out: remove phrase[i] entirely, re-run, measure divergence.
 */
async function saliencyOmission(phrases, baseline, callPerturbed, onTick) {
  const scores = [];
  for (let i = 0; i < phrases.length; i++) {
    const omitted = phrases.filter((_, j) => j !== i).join('') || ' ';
    try {
      const out = await callPerturbed(omitted);
      scores.push(1 - cosineSim(baseline, out));
    } catch {
      scores.push(0);
    }
    onTick(i + 1, phrases.length);
  }
  return scores;
}

/**
 * Paraphrase: ask the model to neutralise phrase[i], then re-run, measure divergence.
 * Costs 2× the API calls per phrase.
 */
async function saliencyParaphrase(phrases, baseline, callPerturbed, onTick) {
  const scores = [];
  for (let i = 0; i < phrases.length; i++) {
    let neutral = '[something]';
    try {
      neutral = await callLLM(
        `Rewrite the following phrase to remove all specific information, ` +
        `making it maximally vague and uninformative. ` +
        `Keep roughly the same character length. ` +
        `Return ONLY the rewritten phrase — no explanation, no quotes:\n${phrases[i]}`,
        '',
        60
      );
    } catch { /* fall back to default neutral */ }

    const perturbed = phrases.map((p, j) => (j === i ? neutral : p)).join('');
    try {
      const out = await callPerturbed(perturbed);
      scores.push(1 - cosineSim(baseline, out));
    } catch {
      scores.push(0);
    }
    onTick(i + 1, phrases.length);
  }
  return scores;
}

// ─── Score normalisation ──────────────────────────────────────────────────────

function normalise(scores) {
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  if (max === min) return scores.map(() => 0.5);
  return scores.map(s => (s - min) / (max - min));
}

// ─── Colour mapping ───────────────────────────────────────────────────────────

/**
 * Map a normalised score [0,1] to an RGBA background colour.
 * Gradient: dark-blue (0) → cyan-blue (0.25) → amber (0.5) → orange (0.75) → red (1)
 */
function scoreToBackground(s) {
  const stops = [
    { t: 0.00, r:  15, g:  40, b:  80, a: 0.28 },
    { t: 0.25, r:  20, g:  80, b: 160, a: 0.45 },
    { t: 0.50, r: 185, g: 130, b:   0, a: 0.55 },
    { t: 0.75, r: 235, g:  70, b:   0, a: 0.70 },
    { t: 1.00, r: 255, g:  25, b:   0, a: 0.88 },
  ];
  let lo = stops[0], hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (s >= stops[i].t && s <= stops[i + 1].t) { lo = stops[i]; hi = stops[i + 1]; break; }
  }
  const t = (hi.t === lo.t) ? 0 : (s - lo.t) / (hi.t - lo.t);
  const lerp = (a, b) => Math.round(a + t * (b - a));
  return `rgba(${lerp(lo.r, hi.r)},${lerp(lo.g, hi.g)},${lerp(lo.b, hi.b)},${(lo.a + t * (hi.a - lo.a)).toFixed(2)})`;
}

function scoreToBorder(s) {
  if (s < 0.3) return 'transparent';
  const alpha = (s * 0.55).toFixed(2);
  const g = Math.round(80 - s * 70);
  return `rgba(255,${g},0,${alpha})`;
}

// ─── Render ───────────────────────────────────────────────────────────────────

function renderSaliency(phrases, normScores) {
  const container = document.getElementById('saliencyText');
  container.innerHTML = '';

  phrases.forEach((phrase, i) => {
    const s   = normScores[i];
    const pct = Math.round(s * 100);
    const span = document.createElement('span');
    span.className = 'phrase fade-in';
    span.textContent = phrase;
    span.style.cssText = `
      background: ${scoreToBackground(s)};
      box-shadow: inset 0 0 0 1px ${scoreToBorder(s)};
      animation-delay: ${i * 35}ms;
    `;
    span.setAttribute('data-tip', `${pct}% impact`);
    container.appendChild(span);
  });
}

function renderStats(phrases, normScores) {
  document.getElementById('statsRow').style.display = 'grid';

  document.getElementById('statPhrases').textContent = phrases.length;

  const maxIdx = normScores.indexOf(Math.max(...normScores));
  const top = phrases[maxIdx].trim();
  document.getElementById('statTop').textContent =
    top.length > 32 ? top.slice(0, 30) + '…' : top;

  const lowCount = normScores.filter(s => s < 0.25).length;
  document.getElementById('statRedundancy').textContent =
    `${Math.round((lowCount / phrases.length) * 100)}%`;
}

// ─── Progress helpers ─────────────────────────────────────────────────────────

function setProgress(msg, pct) {
  document.getElementById('phaseLog').textContent = `▶ ${msg}`;
  document.getElementById('progressBar').style.width = `${pct}%`;
}

function showProgressUI(show) {
  const wrap = document.getElementById('progressWrap');
  const log  = document.getElementById('phaseLog');
  wrap.className = show ? 'progress-wrap active' : 'progress-wrap';
  log.className  = show ? 'phase-log active' : 'phase-log';
}

// ─── Error / UI helpers ───────────────────────────────────────────────────────

function showError(msg) {
  const el = document.getElementById('errorBanner');
  el.innerHTML = `⚠ ${msg}`;
  el.classList.add('show');
}
function hideError() {
  document.getElementById('errorBanner').classList.remove('show');
}

function setRunning(running) {
  const btn = document.getElementById('runBtn');
  btn.disabled = running;
  btn.classList.toggle('loading', running);
  document.getElementById('btnLabel').textContent = running ? 'Analyzing…' : 'Analyze Prompt';
}

// ─── Main analysis ────────────────────────────────────────────────────────────

async function runAnalysis() {
  const userText   = document.getElementById('prompt').value.trim();
  const systemText = document.getElementById('systemprompt').value.trim();

  // Route: whichever is the analysis target is "primary"; the other is fixed context
  const analyzingUser = analysisTarget === 'user';
  const primaryText   = analyzingUser ? userText   : systemText;
  const contextText   = analyzingUser ? systemText : userText;

  // For the API call: system prompt is always the system role,
  // user prompt is always the user role — regardless of which we're analyzing.
  // When analyzing the system prompt, contextText goes in the user role and
  // primaryText is what we perturb in the system role.
  const buildCall = (analyzedText) => ({
    userMsg:   analyzingUser ? analyzedText : contextText,
    systemMsg: analyzingUser ? contextText  : analyzedText,
  });

  hideError();

  if (!getKey()) {
    openKeyModal();
    return;
  }
  if (!primaryText) {
    const label = analyzingUser ? 'user prompt' : 'system prompt';
    showError(`Please enter a ${label} to analyze.`);
    return;
  }

  setRunning(true);
  document.getElementById('statsRow').style.display = 'none';
  document.getElementById('modelOutputCard').style.display = 'none';
  document.getElementById('saliencyText').innerHTML = `
    <div class="empty-state">
      <div class="empty-icon" style="border-color:var(--accent);animation:spin 1.4s linear infinite">◎</div>
      <p>Running saliency analysis…</p>
    </div>`;

  showProgressUI(true);
  setProgress('Tokenizing prompt…', 5);

  try {
    // 1. Tokenize the primary (analyzed) prompt
    const phrases = tokenizePhrases(primaryText);
    document.getElementById('phraseCount').textContent =
      `${phrases.length} phrase${phrases.length !== 1 ? 's' : ''}`;

    // 2. Baseline — full unperturbed primary prompt
    setProgress(`Getting baseline response (${phrases.length} phrases found)…`, 15);
    const { userMsg: baseUser, systemMsg: baseSys } = buildCall(primaryText);
    const baseline = await callLLM(baseUser, baseSys, 600);

    document.getElementById('modelOutputCard').style.display = 'block';
    document.getElementById('modelResponse').textContent = baseline;

    // 3. Saliency — perturb one phrase at a time from the primary prompt
    setProgress(`Running ${selectedMethod} saliency…`, 20);
    const onTick = (done, total) => {
      const pct = 20 + Math.round((done / total) * 72);
      setProgress(`${selectedMethod}: ${done} / ${total} phrases…`, pct);
    };

    // Wrap saliency methods to use buildCall routing
    const perturbedCall = async (perturbedText) => {
      const { userMsg, systemMsg } = buildCall(perturbedText);
      return callLLM(userMsg, systemMsg, 400);
    };

    let rawScores;
    if (selectedMethod === 'perturbation') {
      rawScores = await saliencyPerturbation(phrases, baseline, perturbedCall, onTick);
    } else if (selectedMethod === 'omission') {
      rawScores = await saliencyOmission(phrases, baseline, perturbedCall, onTick);
    } else {
      rawScores = await saliencyParaphrase(phrases, baseline, perturbedCall, onTick);
    }

    // 4. Normalise and render
    setProgress('Normalizing scores and rendering…', 95);
    const normScores = normalise(rawScores);
    renderSaliency(phrases, normScores);
    renderStats(phrases, normScores);

    setProgress('Analysis complete ✓', 100);
    setTimeout(() => showProgressUI(false), 1400);

  } catch (err) {
    showError(err.message || 'Unexpected error. Open DevTools console for details.');
    showProgressUI(false);
    document.getElementById('saliencyText').innerHTML =
      `<div class="empty-state"><p style="color:var(--warn)">Analysis failed. See the error banner above.</p></div>`;
  } finally {
    setRunning(false);
  }
}

// ─── Keyboard shortcut ────────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runAnalysis();
  if (e.key === 'Escape') closeKeyModal();
});

// ─── Init ─────────────────────────────────────────────────────────────────────

updateKeyButton();
