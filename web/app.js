/**
 * PromptLens — app.js
 *
 * Architecture:
 *  1. User selects a provider (Groq or Together AI) and supplies their API key,
 *     stored in localStorage — never sent anywhere except the chosen provider's servers.
 *  2. The prompt is classified and segmented into structural phrases.
 *  3. A baseline response is obtained from the selected model.
 *  4. Monte Carlo Shapley Attribution computes an importance score per phrase by
 *     averaging each phrase's marginal contribution across M random coalition walks.
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
    rateLimitMsg:    'Together AI rate limit hit. Wait a moment then try again.',
    embeddingModel:  'nomic-ai/nomic-embed-text-v1.5',
    judgeModel:      'mistralai/Mixtral-8x7B-Instruct-v0.1',
  },
};

// ─── Constants ────────────────────────────────────────────────────────────────

const SimilarityStrategy = {
  TRIGRAM:  'trigram',   // character trigram cosine — always available
  SEMANTIC: 'semantic',  // embedding + LLM judge — requires Together AI key
};

const SIM_DESCRIPTIONS = {
  trigram:  'Compares outputs using character trigram overlap. Fast with no extra API calls, but scores surface rewording as divergence even when the meaning is unchanged.',
  semantic: 'Embeds outputs to measure meaning-level divergence across Shapley walks. More accurate signal for semantic rewording — uses additional Together AI embedding calls.',
};

// ─── State ────────────────────────────────────────────────────────────────────

let shapleyM           = 20;
let selectedProvider   = localStorage.getItem('promptlens_provider') || 'groq';
let selectedSimilarity = SimilarityStrategy.SEMANTIC;

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
  updateSimilaritySelector();
  closeKeyModal();
}

function clearKey() {
  const p = PROVIDERS[selectedProvider];
  localStorage.removeItem(p.storageKey);
  document.getElementById('keyInput').value = '';
  document.getElementById('keyError').textContent = '';
  updateKeyButton();
  updateSimilaritySelector();
}

// ─── Method selector ──────────────────────────────────────────────────────────

function selectShapleyM(btn) {
  document.querySelectorAll('#advancedRow .method-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  shapleyM = parseInt(btn.dataset.m, 10);
}

function toggleAdvanced() {
  const row = document.getElementById('advancedRow');
  const btn = document.querySelector('.advanced-toggle');
  const open = row.style.display === 'none' || row.style.display === '';
  row.style.display = open ? 'flex' : 'none';
  btn.textContent = open ? 'Advanced ▴' : 'Advanced ▾';
}

function selectSimilarity(btn) {
  document.querySelectorAll('.sim-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedSimilarity = btn.dataset.sim;
}

/** Show/hide the similarity selector depending on whether a Together AI key exists. */
function updateSimilaritySelector() {
  const hasTogetherKey = !!getKey('together');
  document.getElementById('similarityRow').style.display = hasTogetherKey ? 'flex' : 'none';
  if (!hasTogetherKey) selectedSimilarity = SimilarityStrategy.TRIGRAM;
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

// ─── Region detection ────────────────────────────────────────────────────────

/**
 * Scan the prompt once and return an ordered array of structural regions.
 * Each region has { type, text } where type is one of:
 *   'plain' | 'code_block' | 'json' | 'bullets' | 'xml_tagged' | 'markdown'
 */
function detectRegions(prompt) {
  const lines  = prompt.split('\n');
  const regions = [];
  let bufLines = [];
  let bufType  = 'plain';

  const flush = () => {
    const text = bufLines.join('\n');
    if (text.trim()) regions.push({ type: bufType, text });
    bufLines = [];
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // ── Fenced code block ──────────────────────────────────────────────
    if (/^\s*```/.test(line)) {
      flush();
      bufType = 'code_block';
      bufLines.push(line);
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) bufLines.push(lines[i++]);
      if (i < lines.length) bufLines.push(lines[i++]); // closing ```
      flush();
      bufType = 'plain';
      continue;
    }

    // ── Block-level JSON (line starts with { or [) ─────────────────────
    if (/^\s*[{[]/.test(line)) {
      flush();
      bufType = 'json';
      let depth = 0;
      while (i < lines.length) {
        const l = lines[i];
        depth += (l.match(/[{[]/g) || []).length - (l.match(/[}\]]/g) || []).length;
        bufLines.push(l);
        i++;
        if (depth <= 0) break;
      }
      flush();
      bufType = 'plain';
      continue;
    }

    // ── XML tagged block (line opens with a tag) ────────────────────────
    if (/^<([a-zA-Z]\w*)>/.test(line.trim())) {
      flush();
      bufType = 'xml_tagged';
      // Track tag depth so multi-line <tag>\ncontent\n</tag> is collected whole
      let tagDepth = 0;
      while (i < lines.length) {
        const l = lines[i];
        tagDepth += (l.match(/<[a-zA-Z]\w*>/g)  || []).length;
        tagDepth -= (l.match(/<\/[a-zA-Z]\w*>/g) || []).length;
        bufLines.push(l);
        i++;
        if (tagDepth <= 0) break;
      }
      flush();
      bufType = 'plain';
      continue;
    }

    // ── Markdown header (# / ## / ###) ────────────────────────────────
    if (/^#{1,3}\s/.test(line)) {
      flush();
      bufType = 'markdown';
      bufLines.push(line);
      i++;
      while (i < lines.length &&
             !/^#{1,3}\s/.test(lines[i]) &&
             !/^\s*```/.test(lines[i]) &&
             !/^\s*[{[]/.test(lines[i]) &&
             !/^<([a-zA-Z]\w*)>/.test(lines[i].trim())) {
        bufLines.push(lines[i++]);
      }
      flush();
      bufType = 'plain';
      continue;
    }

    // ── Bullet list (-, *, •, or N.) ──────────────────────────────────
    if (/^\s*[-*•]/.test(line) || /^\s*\d+\.\s/.test(line)) {
      flush();
      bufType = 'bullets';
      while (i < lines.length) {
        const l     = lines[i];
        const isBullet       = /^\s*[-*•]/.test(l) || /^\s*\d+\.\s/.test(l);
        const isContinuation = /^\s{2,}/.test(l) && l.trim() && !isBullet;
        const isEmpty        = !l.trim();
        if (!isBullet && !isContinuation && !isEmpty) break;
        if (isEmpty) {
          // Keep the empty line only if the next non-empty line is another bullet
          const next = lines.slice(i + 1).find(x => x.trim());
          if (!next || (!/^\s*[-*•]/.test(next) && !/^\s*\d+\.\s/.test(next))) break;
        }
        bufLines.push(l);
        i++;
      }
      flush();
      bufType = 'plain';
      continue;
    }

    // ── Plain text ─────────────────────────────────────────────────────
    bufLines.push(line);
    i++;
  }

  flush();
  return regions;
}

// ─── Per-type segmenters ──────────────────────────────────────────────────────
//
// Each returns Phrase[]: { text, atomic, tagName?, content? }
// Phrase.atomic = true  → never split further by character count
// Phrase.tagName        → XML tag name, used to reconstruct the wrapper during perturbation

/** Plain prose — existing sentence/clause heuristic, unchanged in behaviour. */
function segmentPlainText(text) {
  const sentRe = /([^.!?\n]+[.!?\n]+)|([^.!?\n]+$)/g;
  const phrases = [];
  let m;
  while ((m = sentRe.exec(text)) !== null) {
    const s = m[0];
    if (!s.trim()) continue;
    if (s.length > 60) {
      const parts = s.split(/(?<=[,;])/);
      let acc = '';
      for (let j = 0; j < parts.length; j++) {
        acc += parts[j];
        if (acc.trim().length >= 35 || j === parts.length - 1) {
          if (acc.trim()) phrases.push({ text: acc, atomic: false });
          acc = '';
        }
      }
      if (acc.trim()) phrases.push({ text: acc, atomic: false });
    } else {
      phrases.push({ text: s, atomic: false });
    }
  }
  return phrases.length > 0 ? phrases : [{ text, atomic: false }];
}

/** Fenced code block — always treated as a single atomic phrase. */
function segmentCodeBlock(text) {
  return [{ text, atomic: true }];
}

/** Bullet list — one phrase per bullet item (with multi-line continuation support). */
function segmentBullets(text) {
  const lines   = text.split('\n');
  const phrases = [];
  let current   = null;
  for (const line of lines) {
    if (!line.trim()) continue;
    const isBullet = /^\s*[-*•]/.test(line) || /^\s*\d+\.\s/.test(line);
    if (isBullet) {
      if (current !== null) phrases.push({ text: current.trim(), atomic: false });
      current = line;
    } else if (current !== null && /^\s{2,}/.test(line)) {
      current += '\n' + line; // continuation of previous bullet
    }
  }
  if (current !== null) phrases.push({ text: current.trim(), atomic: false });
  return phrases;
}

/** JSON object — one phrase per top-level key. Arrays, empty objects, and malformed JSON → atomic. */
function segmentJSON(text) {
  try {
    const parsed = JSON.parse(text.trim());
    if (Array.isArray(parsed)) return [{ text, atomic: true }];
    const entries = Object.entries(parsed);
    if (entries.length === 0) return [{ text, atomic: true }];
    return entries.map(([key, value]) => ({
      text:   `"${key}": ${JSON.stringify(value)}`,
      atomic: false,
    }));
  } catch {
    return [{ text, atomic: true }];
  }
}

/** XML tags — one phrase per outermost tag, metadata preserved for perturbation. */
function segmentXML(text) {
  const tagPattern = /<([a-zA-Z]\w*)>([\s\S]*?)<\/\1>/g;
  const phrases    = [];
  let match;
  while ((match = tagPattern.exec(text)) !== null) {
    phrases.push({
      text:    match[0],
      tagName: match[1],
      content: match[2].trim(),
      atomic:  false,
    });
  }
  return phrases.length > 0 ? phrases : segmentPlainText(text);
}

/** Markdown — headers are atomic anchors; body within each section uses plain segmenter. */
function segmentMarkdown(text) {
  const sections = text.split(/(?=^#{1,3}\s)/m);
  return sections.flatMap(section => {
    const headerMatch = section.match(/^(#{1,3}\s.+)\n?/);
    if (headerMatch) {
      const headerPhrase = { text: headerMatch[1].trim(), atomic: true };
      const body = section.slice(headerMatch[0].length);
      return body.trim() ? [headerPhrase, ...segmentPlainText(body)] : [headerPhrase];
    }
    return segmentPlainText(section);
  });
}

/**
 * Main entry point — classify-then-segment pipeline.
 * Returns Phrase[] representing all phrase units in the prompt.
 */
function segmentPrompt(prompt) {
  const regions = detectRegions(prompt);
  return regions.flatMap(region => {
    switch (region.type) {
      case 'code_block': return segmentCodeBlock(region.text);
      case 'json':       return segmentJSON(region.text);
      case 'bullets':    return segmentBullets(region.text);
      case 'xml_tagged': return segmentXML(region.text);
      case 'markdown':   return segmentMarkdown(region.text);
      default:           return segmentPlainText(region.text);
    }
  });
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

// ─── Embedding & semantic similarity ─────────────────────────────────────────

async function getEmbedding(text) {
  const p   = PROVIDERS.together;
  const key = getKey('together');
  const embeddingEndpoint = p.endpoint.replace('/chat/completions', '/embeddings');
  const res = await fetch(embeddingEndpoint, {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${key}`,
    },
    body: JSON.stringify({ model: p.embeddingModel, input: text }),
  });
  if (!res.ok) throw new Error(`Embedding API error: HTTP ${res.status}`);
  const data = await res.json();
  return data.data[0].embedding;
}

function vecCosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

/** Run up to `limit` async tasks concurrently. taskFns is an array of () => Promise. */
async function withConcurrency(limit, taskFns) {
  const results = new Array(taskFns.length);
  let next = 0;
  async function worker() {
    while (next < taskFns.length) {
      const i = next++;
      results[i] = await taskFns[i]();
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, taskFns.length) }, worker));
  return results;
}

// ─── Shapley Attribution ──────────────────────────────────────────────────────

/** Fisher-Yates shuffle — returns a new shuffled array. */
function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Generate all N! permutations of [0, 1, …, N-1]. Used for exact Shapley when N ≤ 4. */
function getAllPermutations(n) {
  const result = [];
  const arr    = [...Array(n).keys()];
  function permute(current, remaining) {
    if (remaining.length === 0) { result.push(current); return; }
    for (let i = 0; i < remaining.length; i++) {
      permute(
        [...current, remaining[i]],
        [...remaining.slice(0, i), ...remaining.slice(i + 1)]
      );
    }
  }
  permute([], arr);
  return result;
}

/**
 * Build a prompt string from the active subset of phrases.
 * Absent XML-tagged phrases become <tag>[...]</tag> (preserves document structure).
 * Absent plain phrases are omitted entirely.
 */
function buildPromptFromCoalition(phrases, activeIndices) {
  const parts = phrases.map((p, i) => {
    if (activeIndices.has(i)) return p.text;
    return p.tagName ? `<${p.tagName}>[...]</${p.tagName}>` : '';
  });
  return parts.join('') || ' ';
}

/**
 * Returns a runner with an LLM coalition cache and (optionally) an embedding cache.
 * Cache key = sorted active indices, so identical subsets share a result.
 */
function makeCoalitionRunner(phrases, callPrompt) {
  const llmCache = new Map();
  const embCache = new Map();

  async function run(activeIndices) {
    const key = [...activeIndices].sort((a, b) => a - b).join(',');
    if (!llmCache.has(key)) {
      const text = buildPromptFromCoalition(phrases, activeIndices);
      llmCache.set(key, callPrompt(text));   // store Promise — deduplicates concurrent calls
    }
    return llmCache.get(key);
  }

  async function embed(output) {
    if (!embCache.has(output)) {
      embCache.set(output, getEmbedding(output));
    }
    return embCache.get(output);
  }

  return { run, embed };
}

/**
 * One Shapley walk along a phrase ordering (random or fixed).
 * Returns marginal contributions[], one per phrase index.
 */
async function sampleShapleyWalk(phrases, runner, emptyOutput, emptyEmb, isSemantic, fixedOrder = null) {
  const N     = phrases.length;
  const order = fixedOrder || shuffleArray([...Array(N).keys()]);
  const contributions = new Array(N).fill(0);

  let activeIndices = new Set();
  let currentOutput = emptyOutput;
  let currentEmb    = emptyEmb;    // null when not in semantic mode

  for (const i of order) {
    activeIndices = new Set([...activeIndices, i]);
    const newOutput = await runner.run(activeIndices);

    let marginal;
    if (isSemantic && currentEmb !== null) {
      const newEmb = await runner.embed(newOutput);
      marginal  = Math.max(0, 1 - vecCosineSim(currentEmb, newEmb));
      currentEmb = newEmb;
    } else {
      marginal = Math.max(0, 1 - cosineSim(currentOutput, newOutput));
    }

    contributions[i] = marginal;
    currentOutput    = newOutput;
  }

  return contributions;
}

/**
 * Main Shapley computation.
 * - N ≤ 4  → exact (all N! permutations, shared coalition cache)
 * - N > 4  → Monte Carlo (shapleyM random walks, concurrency cap 5)
 *
 * Returns raw (un-normalised) Shapley scores, one per phrase.
 */
async function computeShapley(phrases, isSemantic, callPrompt, onTick) {
  const N = phrases.length;

  // Empty coalition — floor reference
  setProgress('Getting empty coalition baseline…', 18);
  const emptyOutput = await callPrompt(buildPromptFromCoalition(phrases, new Set()));

  // Pre-compute empty-coalition embedding for semantic mode
  let emptyEmb = null;
  if (isSemantic) {
    setProgress('Embedding empty coalition…', 20);
    try { emptyEmb = await getEmbedding(emptyOutput); } catch (e) {
      console.warn('Empty coalition embedding failed, falling back to trigram:', e);
    }
  }

  const runner = makeCoalitionRunner(phrases, callPrompt);

  // Determine which orderings to walk
  const orderings = N <= 4 ? getAllPermutations(N) : Array.from({ length: shapleyM }, () => null);
  const totalWalks = orderings.length;
  let completedWalks = 0;

  const taskFns = orderings.map(fixedOrder => async () => {
    const result = await sampleShapleyWalk(
      phrases, runner, emptyOutput, emptyEmb, isSemantic, fixedOrder
    );
    completedWalks++;
    onTick(completedWalks, totalWalks);
    return result;
  });

  const allContributions = await withConcurrency(5, taskFns);

  return phrases.map((_, i) =>
    allContributions.reduce((sum, c) => sum + c[i], 0) / totalWalks
  );
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
    span.textContent = phrase.text;
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
  const top = phrases[maxIdx].text.trim();
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
  // #systemprompt is always the analysis subject; #prompt is the held-constant test input.
  const systemText = document.getElementById('systemprompt').value.trim();
  const userText   = document.getElementById('prompt').value.trim();

  const primaryText = systemText;
  const contextText = userText;

  // System prompt is perturbed; user message is held constant as context.
  const buildCall = (analyzedText) => ({
    userMsg:   contextText,
    systemMsg: analyzedText,
  });

  hideError();

  if (!getKey()) {
    openKeyModal();
    return;
  }
  if (!primaryText) {
    showError('Please enter a system prompt to analyze.');
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
  setProgress('Segmenting prompt…', 5);

  try {
    // 1. Segment the primary (analyzed) prompt into structural phrases
    const phrases = segmentPrompt(primaryText);
    document.getElementById('phraseCount').textContent =
      `${phrases.length} phrase${phrases.length !== 1 ? 's' : ''}`;

    // 2. Baseline — full unperturbed primary prompt
    setProgress(`Getting baseline response (${phrases.length} phrases found)…`, 15);
    const { userMsg: baseUser, systemMsg: baseSys } = buildCall(primaryText);
    const baseline = await callLLM(baseUser, baseSys, 600);

    document.getElementById('modelOutputCard').style.display = 'block';
    document.getElementById('modelResponse').textContent = baseline;

    // 3. Shapley Attribution
    const isSemantic = selectedSimilarity === SimilarityStrategy.SEMANTIC && !!getKey('together');
    const onTick = (done, total) => {
      const pct = 22 + Math.round((done / total) * 70);
      setProgress(`Shapley: ${done} / ${total} walks…`, pct);
    };

    // Wrap coalition runner to use buildCall routing
    const coalitionCall = async (coalitionText) => {
      const { userMsg, systemMsg } = buildCall(coalitionText);
      return callLLM(userMsg, systemMsg, 400);
    };

    // 4. Compute Shapley scores
    const rawScores = await computeShapley(phrases, isSemantic, coalitionCall, onTick);

    // 5. Normalise and render
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
updateSimilaritySelector();
