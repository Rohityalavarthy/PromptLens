/**
 * PromptLens — app.js
 *
 * Architecture:
 *  1. User selects a provider and supplies their API key (stored in localStorage).
 *  2. The prompt is classified and segmented into structural phrases.
 *  3. A baseline response is obtained from the selected model.
 *  4. Monte Carlo Shapley Attribution computes an importance score per phrase.
 *  5. Scores are min-max normalised and rendered as inline colour spans.
 */

'use strict';

// ─── Provider config ──────────────────────────────────────────────────────────

const PROVIDERS = {
  groq: {
    id:          'groq',
    label:       'Groq',
    model:       'llama-3.3-70b-versatile',
    endpoint:    'https://api.groq.com/openai/v1/chat/completions',
    storageKey:  'promptlens_groq_key',
    keyPrefix:   'gsk_',
    keyHint:     'gsk_••••••••••••••••••••••••••••••',
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
    keyPrefix:   null,
    keyHint:     '••••••••••••••••••••••••••••••••',
    signupUrl:   'https://api.together.ai/settings/api-keys',
    signupLabel: 'api.together.ai/settings/api-keys',
    freeNote:    'Free $1 credit on sign-up · No credit card required',
    rateLimitMsg:    'Together AI rate limit hit. Wait a moment then try again.',
    embeddingModel:  'nomic-ai/nomic-embed-text-v1.5',
    judgeModel:      'mistralai/Mixtral-8x7B-Instruct-v0.1',
  },
  openai: {
    id:          'openai',
    label:       'OpenAI',
    model:       'gpt-4o-mini',
    endpoint:    'https://api.openai.com/v1/chat/completions',
    storageKey:  'promptlens_openai_key',
    keyPrefix:   'sk-',
    keyHint:     'sk-••••••••••••••••••••••••',
    signupUrl:   'https://platform.openai.com/api-keys',
    signupLabel: 'platform.openai.com/api-keys',
    freeNote:    'Pay-per-use · No free tier · Fast',
    rateLimitMsg:'OpenAI rate limit hit. Wait a moment then try again.',
    embeddingModel: 'text-embedding-3-small',
  },
  anthropic: {
    id:          'anthropic',
    label:       'Anthropic',
    model:       'claude-haiku-4-5-20251001',
    endpoint:    'https://api.anthropic.com/v1/messages',
    storageKey:  'promptlens_anthropic_key',
    keyPrefix:   'sk-ant-',
    keyHint:     'sk-ant-••••••••••••••••••••',
    signupUrl:   'https://console.anthropic.com/settings/api-keys',
    signupLabel: 'console.anthropic.com/settings/api-keys',
    freeNote:    'Pay-per-use · No free tier',
    rateLimitMsg:'Anthropic rate limit hit. Wait a moment then try again.',
  },
};

// ─── Constants ────────────────────────────────────────────────────────────────

const SimilarityStrategy = {
  TRIGRAM:  'trigram',
  SEMANTIC: 'semantic',
};

const SIM_DESCRIPTIONS = {
  trigram:  'Compares outputs using character trigram overlap. Fast with no extra API calls, but scores surface rewording as divergence even when the meaning is unchanged.',
  semantic: 'Embeds outputs to measure meaning-level divergence across Shapley walks. More accurate signal for semantic rewording — uses additional embedding API calls (Together AI or OpenAI).',
};

// ─── State ────────────────────────────────────────────────────────────────────

let shapleyM           = 20;
let selectedProvider   = localStorage.getItem('promptlens_provider') || 'groq';
let selectedSimilarity = SimilarityStrategy.SEMANTIC;
let lastPhrases        = null;
let lastNormScores     = null;
let viewMode           = 'heatmap';
let scoreThreshold     = 0;
let tableSortCol       = 'score';
let tableSortDir       = -1;
let lastCompressDiff    = null;
let lastOriginalText    = null;
let compressThreshold   = 0.15;

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

function refreshModalForProvider(providerId) {
  const p = PROVIDERS[providerId];

  document.querySelectorAll('.provider-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.provider === providerId);
  });

  document.getElementById('providerSignupUrl').href         = p.signupUrl;
  document.getElementById('providerSignupLabel').textContent = p.signupLabel;
  document.getElementById('providerFreeNote').textContent   = p.freeNote;
  document.getElementById('providerModel').textContent      = p.model;

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
  const row  = document.getElementById('advancedRow');
  const btn  = document.querySelector('.advanced-toggle');
  const open = row.style.display === 'none' || row.style.display === '';
  row.style.display = open ? 'flex' : 'none';
  btn.textContent = open ? 'Advanced ▴' : 'Advanced ▾';
}

function selectSimilarity(btn) {
  document.querySelectorAll('.sim-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedSimilarity = btn.dataset.sim;
}

/** Show similarity selector when Together AI or OpenAI key is present (both support embeddings). */
function updateSimilaritySelector() {
  const hasEmbeddingKey = !!getKey('together') || !!getKey('openai');
  document.getElementById('similarityRow').style.display = hasEmbeddingKey ? 'flex' : 'none';
  if (!hasEmbeddingKey) selectedSimilarity = SimilarityStrategy.TRIGRAM;
}

// ─── LLM calls ────────────────────────────────────────────────────────────────

/**
 * Call the active provider's chat completions endpoint.
 * Groq, Together AI, and OpenAI use the OpenAI-compatible format.
 * Anthropic uses a separate code path.
 */
async function callLLM(userMsg, systemMsg = '', maxTokens = 500) {
  const p   = getProvider();
  const key = getKey();

  if (!key) {
    throw new Error(`No ${p.label} API key set. Click "Add API Key" to add your key.`);
  }

  if (p.id === 'anthropic') {
    return callAnthropicLLM(userMsg, systemMsg, maxTokens, p, key);
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
      temperature: 0.0,
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

async function callAnthropicLLM(userMsg, systemMsg, maxTokens, p, key) {
  const body = {
    model:      p.model,
    max_tokens: maxTokens,
    messages:   [{ role: 'user', content: userMsg }],
  };
  if (systemMsg) body.system = systemMsg;

  const res = await fetch(p.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         key,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = err?.error?.message || `HTTP ${res.status}`;
    if (res.status === 401) throw new Error(`Invalid ${p.label} API key. Check your key and try again.`);
    if (res.status === 429) throw new Error(p.rateLimitMsg);
    throw new Error(`${p.label} API error: ${msg}`);
  }

  const data = await res.json();
  return data.content?.[0]?.text ?? '';
}

// ─── Region detection ─────────────────────────────────────────────────────────

/**
 * Scan the prompt once and return an ordered array of structural regions.
 * Each region has { type, text } where type is one of:
 *   'plain' | 'code_block' | 'json' | 'bullets' | 'xml_tagged' | 'markdown'
 */
function detectRegions(prompt) {
  const lines   = prompt.split('\n');
  const regions = [];
  let bufLines  = [];
  let bufType   = 'plain';

  const flush = () => {
    const text = bufLines.join('\n');
    if (text.trim()) regions.push({ type: bufType, text });
    bufLines = [];
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    if (/^\s*```/.test(line)) {
      flush();
      bufType = 'code_block';
      bufLines.push(line);
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) bufLines.push(lines[i++]);
      if (i < lines.length) bufLines.push(lines[i++]);
      flush();
      bufType = 'plain';
      continue;
    }

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

    if (/^<([a-zA-Z]\w*)>/.test(line.trim())) {
      flush();
      bufType = 'xml_tagged';
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

    if (/^\s*[-*•]/.test(line) || /^\s*\d+\.\s/.test(line)) {
      flush();
      bufType = 'bullets';
      while (i < lines.length) {
        const l            = lines[i];
        const isBullet     = /^\s*[-*•]/.test(l) || /^\s*\d+\.\s/.test(l);
        const isContinuation = /^\s{2,}/.test(l) && l.trim() && !isBullet;
        const isEmpty      = !l.trim();
        if (!isBullet && !isContinuation && !isEmpty) break;
        if (isEmpty) {
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

    bufLines.push(line);
    i++;
  }

  flush();
  return regions;
}

// ─── Per-type segmenters ──────────────────────────────────────────────────────

function segmentPlainText(text) {
  const sentRe  = /([^.!?\n]+[.!?\n]+)|([^.!?\n]+$)/g;
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

function segmentCodeBlock(text) {
  return [{ text, atomic: true }];
}

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
      current += '\n' + line;
    }
  }
  if (current !== null) phrases.push({ text: current.trim(), atomic: false });
  return phrases;
}

function segmentJSON(text) {
  try {
    const parsed  = JSON.parse(text.trim());
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
 * Attaches regionType to each phrase for table display.
 */
function segmentPrompt(prompt) {
  const regions = detectRegions(prompt);
  return regions.flatMap(region => {
    let phrases;
    switch (region.type) {
      case 'code_block': phrases = segmentCodeBlock(region.text); break;
      case 'json':       phrases = segmentJSON(region.text);      break;
      case 'bullets':    phrases = segmentBullets(region.text);   break;
      case 'xml_tagged': phrases = segmentXML(region.text);       break;
      case 'markdown':   phrases = segmentMarkdown(region.text);  break;
      default:           phrases = segmentPlainText(region.text); break;
    }
    return phrases.map(p => ({ ...p, regionType: region.type }));
  });
}

// ─── Similarity ───────────────────────────────────────────────────────────────

function ngramFreq(text, n = 3) {
  const v = {};
  const t = text.toLowerCase();
  for (let i = 0; i <= t.length - n; i++) {
    const g = t.slice(i, i + n);
    v[g] = (v[g] || 0) + 1;
  }
  return v;
}

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

/**
 * Fetch an embedding vector. Prefers Together AI; falls back to OpenAI.
 * Anthropic has no embedding endpoint — semantic mode falls back to trigram
 * automatically via updateSimilaritySelector().
 */
async function getEmbedding(text) {
  let p, key;
  if (getKey('together')) {
    p = PROVIDERS.together; key = getKey('together');
  } else if (getKey('openai')) {
    p = PROVIDERS.openai; key = getKey('openai');
  } else {
    throw new Error('Semantic similarity requires a Together AI or OpenAI key for embeddings.');
  }

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

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

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

function buildPromptFromCoalition(phrases, activeIndices) {
  const parts = phrases.map((p, i) => {
    if (activeIndices.has(i)) return p.text;
    return p.tagName ? `<${p.tagName}>[...]</${p.tagName}>` : '';
  });
  return parts.join('') || ' ';
}

function makeCoalitionRunner(phrases, callPrompt) {
  const llmCache = new Map();
  const embCache = new Map();

  async function run(activeIndices) {
    const key = [...activeIndices].sort((a, b) => a - b).join(',');
    if (!llmCache.has(key)) {
      const text = buildPromptFromCoalition(phrases, activeIndices);
      llmCache.set(key, callPrompt(text));
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

async function sampleShapleyWalk(phrases, runner, emptyOutput, emptyEmb, isSemantic, fixedOrder = null) {
  const N     = phrases.length;
  const order = fixedOrder || shuffleArray([...Array(N).keys()]);
  const contributions = new Array(N).fill(0);

  let activeIndices = new Set();
  let currentOutput = emptyOutput;
  let currentEmb    = emptyEmb;

  for (const i of order) {
    activeIndices = new Set([...activeIndices, i]);
    const newOutput = await runner.run(activeIndices);

    let marginal;
    if (isSemantic && currentEmb !== null) {
      const newEmb = await runner.embed(newOutput);
      marginal   = Math.max(0, 1 - vecCosineSim(currentEmb, newEmb));
      currentEmb = newEmb;
    } else {
      marginal = Math.max(0, 1 - cosineSim(currentOutput, newOutput));
    }

    contributions[i] = marginal;
    currentOutput    = newOutput;
  }

  return contributions;
}

async function computeShapley(phrases, isSemantic, callPrompt, onTick) {
  const N = phrases.length;

  setProgress('Getting empty coalition baseline…', 18);
  const emptyOutput = await callPrompt(buildPromptFromCoalition(phrases, new Set()));

  let emptyEmb = null;
  if (isSemantic) {
    setProgress('Embedding empty coalition…', 20);
    try { emptyEmb = await getEmbedding(emptyOutput); } catch (e) {
      console.warn('Empty coalition embedding failed, falling back to trigram:', e);
    }
  }

  const runner    = makeCoalitionRunner(phrases, callPrompt);
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

// ─── Utilities ────────────────────────────────────────────────────────────────

function htmlEsc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ─── Render — heatmap ─────────────────────────────────────────────────────────

function renderSaliency(phrases, normScores) {
  lastPhrases    = phrases;
  lastNormScores = normScores;

  if (viewMode === 'table') {
    renderTable(phrases, normScores);
    return;
  }

  const container = document.getElementById('saliencyText');
  container.innerHTML = '';
  container.style.padding = '';

  phrases.forEach((phrase, i) => {
    const s     = normScores[i];
    const pct   = Math.round(s * 100);
    const below = s < scoreThreshold / 100;
    const span  = document.createElement('span');
    span.className   = 'phrase fade-in' + (below ? ' dimmed' : '');
    span.textContent = phrase.text;
    if (!below) {
      span.style.cssText = `
        background: ${scoreToBackground(s)};
        box-shadow: inset 0 0 0 1px ${scoreToBorder(s)};
        animation-delay: ${i * 35}ms;
      `;
    }
    span.setAttribute('data-tip', `${pct}% impact`);
    container.appendChild(span);
  });
}

// ─── Render — table ───────────────────────────────────────────────────────────

function renderTable(phrases, normScores) {
  lastPhrases    = phrases;
  lastNormScores = normScores;

  const rows = phrases.map((p, i) => ({
    phrase: p.text.trim(),
    score:  normScores[i],
    region: p.regionType || 'plain',
    below:  normScores[i] < scoreThreshold / 100,
  }));

  rows.sort((a, b) => {
    if (tableSortCol === 'score')  return tableSortDir * (b.score - a.score);
    if (tableSortCol === 'phrase') return tableSortDir * a.phrase.localeCompare(b.phrase);
    if (tableSortCol === 'region') return tableSortDir * a.region.localeCompare(b.region);
    return 0;
  });

  const arrow = col => tableSortCol === col ? (tableSortDir < 0 ? ' ▾' : ' ▴') : '';

  const container = document.getElementById('saliencyText');
  container.style.padding = '0';
  container.innerHTML = `
    <table class="score-table">
      <thead>
        <tr>
          <th onclick="sortTable('score')">Score${arrow('score')}</th>
          <th onclick="sortTable('phrase')">Phrase${arrow('phrase')}</th>
          <th onclick="sortTable('region')">Region${arrow('region')}</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `
          <tr class="${r.below ? 'row-dimmed' : ''}">
            <td class="td-score">
              <div class="score-bar-wrap">
                <div class="score-bar" style="width:${Math.round(r.score * 72)}px;background:${scoreToBackground(r.score)}"></div>
                <span>${Math.round(r.score * 100)}%</span>
              </div>
            </td>
            <td class="td-phrase">${htmlEsc(r.phrase)}</td>
            <td class="td-region">${r.region}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

function sortTable(col) {
  if (tableSortCol === col) tableSortDir *= -1;
  else { tableSortCol = col; tableSortDir = -1; }
  if (lastPhrases) renderTable(lastPhrases, lastNormScores);
}

function setViewMode(mode) {
  viewMode = mode;
  document.getElementById('viewHeatmapBtn').classList.toggle('active', mode === 'heatmap');
  document.getElementById('viewTableBtn').classList.toggle('active', mode === 'table');
  if (!lastPhrases) return;
  if (mode === 'heatmap') renderSaliency(lastPhrases, lastNormScores);
  else renderTable(lastPhrases, lastNormScores);
}

// ─── Threshold filter ─────────────────────────────────────────────────────────

function applyThreshold(val) {
  scoreThreshold = parseInt(val, 10);
  const label    = document.getElementById('thresholdValue');
  label.textContent = parseInt(val) === 0 ? 'All' : `< ${val}%`;
  if (!lastPhrases) return;
  if (viewMode === 'heatmap') renderSaliency(lastPhrases, lastNormScores);
  else renderTable(lastPhrases, lastNormScores);
}

// ─── Stats ────────────────────────────────────────────────────────────────────

function renderStats(phrases, normScores) {
  document.getElementById('statsRow').style.display = 'grid';

  document.getElementById('statPhrases').textContent = phrases.length;

  const maxIdx = normScores.indexOf(Math.max(...normScores));
  const top    = phrases[maxIdx].text.trim();
  document.getElementById('statTop').textContent =
    top.length > 32 ? top.slice(0, 30) + '…' : top;

  const lowCount = normScores.filter(s => s < 0.25).length;
  document.getElementById('statRedundancy').textContent =
    `${Math.round((lowCount / phrases.length) * 100)}%`;
}

// ─── Copy / Export ────────────────────────────────────────────────────────────

function copyResults() {
  if (!lastPhrases) return;
  const rows = lastPhrases
    .map((p, i) => ({
      phrase: p.text.trim(),
      score:  Math.round(lastNormScores[i] * 100),
      region: p.regionType || 'plain',
    }))
    .sort((a, b) => b.score - a.score);

  const header = '| Score | Phrase | Region |\n|-------|--------|--------|\n';
  const body   = rows.map(r =>
    `| ${r.score}% | ${r.phrase.replace(/\|/g, '\\|')} | ${r.region} |`
  ).join('\n');

  navigator.clipboard.writeText(header + body).then(() => {
    const btn  = document.getElementById('copyBtn');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = orig, 1600);
  });
}

function exportJSON() {
  if (!lastPhrases) return;
  const data = {
    timestamp:  new Date().toISOString(),
    provider:   selectedProvider,
    model:      getProvider().model,
    similarity: selectedSimilarity,
    m_samples:  shapleyM,
    phrases:    lastPhrases.map((p, i) => ({
      text:   p.text,
      score:  parseFloat(lastNormScores[i].toFixed(4)),
      region: p.regionType || 'plain',
      atomic: p.atomic,
    })),
    baseline: document.getElementById('modelResponse').textContent,
  };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `promptlens-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

// ─── Share via URL ────────────────────────────────────────────────────────────

function encodeShareURL() {
  const state = {
    sys: document.getElementById('systemprompt').value,
    usr: document.getElementById('prompt').value,
    p:   selectedProvider,
    m:   shapleyM,
    sim: selectedSimilarity,
  };
  try {
    const encoded = btoa(unescape(encodeURIComponent(JSON.stringify(state))));
    const url     = `${location.origin}${location.pathname}#share=${encoded}`;
    navigator.clipboard.writeText(url).then(() => {
      const btn  = document.getElementById('shareBtn');
      const orig = btn.textContent;
      btn.textContent = 'Link copied!';
      setTimeout(() => btn.textContent = orig, 1800);
    });
  } catch (e) {
    showError('Share failed — prompt may be too large to encode in a URL.');
  }
}

function tryLoadShareURL() {
  const hash = location.hash;
  if (!hash.startsWith('#share=')) return;
  try {
    const encoded = hash.slice(7);
    const state   = JSON.parse(decodeURIComponent(escape(atob(encoded))));
    if (state.sys) document.getElementById('systemprompt').value = state.sys;
    if (state.usr) document.getElementById('prompt').value = state.usr;
    if (state.p && PROVIDERS[state.p]) {
      selectedProvider = state.p;
      localStorage.setItem('promptlens_provider', selectedProvider);
    }
    if (state.m)   shapleyM           = state.m;
    if (state.sim) selectedSimilarity = state.sim;
    history.replaceState(null, '', location.pathname);
  } catch (e) {
    console.warn('Failed to load shared URL:', e);
  }
}

// ─── Run history ──────────────────────────────────────────────────────────────

const HISTORY_KEY = 'promptlens_history';
const HISTORY_MAX = 8;

function saveToHistory(phrases, normScores) {
  const sys = document.getElementById('systemprompt').value;
  const usr = document.getElementById('prompt').value;
  if (!sys.trim()) return;

  const maxIdx   = normScores.indexOf(Math.max(...normScores));
  const lowCount = normScores.filter(s => s < 0.25).length;
  const entry    = {
    ts:          Date.now(),
    sys,
    usr,
    phraseCount: phrases.length,
    redundancy:  Math.round((lowCount / phrases.length) * 100),
    topPhrase:   phrases[maxIdx].text.trim().slice(0, 50),
  };

  const hist = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  hist.unshift(entry);
  if (hist.length > HISTORY_MAX) hist.length = HISTORY_MAX;
  localStorage.setItem(HISTORY_KEY, JSON.stringify(hist));
  renderHistoryBtn();
}

function renderHistoryBtn() {
  const hist = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  document.getElementById('historyWrap').style.display = hist.length > 0 ? '' : 'none';
}

function toggleHistoryDropdown() {
  const dropdown = document.getElementById('historyDropdown');
  if (dropdown.classList.contains('open')) {
    dropdown.classList.remove('open');
    return;
  }
  const hist = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  dropdown.innerHTML = hist.map((entry, idx) => {
    const d    = new Date(entry.ts);
    const date = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const preview = htmlEsc(entry.sys.slice(0, 70)) + (entry.sys.length > 70 ? '…' : '');
    return `<div class="history-entry" onclick="restoreHistory(${idx})">
      <div class="history-meta">${date} ${time} · ${entry.phraseCount} phrases · ${entry.redundancy}% redundant</div>
      <div class="history-preview">${preview}</div>
    </div>`;
  }).join('');
  dropdown.classList.add('open');
}

function restoreHistory(idx) {
  const hist  = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  const entry = hist[idx];
  if (!entry) return;
  document.getElementById('systemprompt').value = entry.sys;
  document.getElementById('prompt').value        = entry.usr || '';
  localStorage.setItem(AUTOSAVE_KEYS.system, entry.sys);
  localStorage.setItem(AUTOSAVE_KEYS.user,   entry.usr || '');
  document.getElementById('historyDropdown').classList.remove('open');
}

// ─── Auto-save ────────────────────────────────────────────────────────────────

const AUTOSAVE_KEYS = {
  system: 'promptlens_autosave_system',
  user:   'promptlens_autosave_user',
};

function initAutoSave() {
  const sys = localStorage.getItem(AUTOSAVE_KEYS.system);
  const usr = localStorage.getItem(AUTOSAVE_KEYS.user);
  if (sys) document.getElementById('systemprompt').value = sys;
  if (usr) document.getElementById('prompt').value = usr;

  let timer;
  const save = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      localStorage.setItem(AUTOSAVE_KEYS.system, document.getElementById('systemprompt').value);
      localStorage.setItem(AUTOSAVE_KEYS.user,   document.getElementById('prompt').value);
    }, 500);
  };
  document.getElementById('systemprompt').addEventListener('input', save);
  document.getElementById('prompt').addEventListener('input', save);
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
  log.className  = show ? 'phase-log active'     : 'phase-log';
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
  const systemText  = document.getElementById('systemprompt').value.trim();
  const userText    = document.getElementById('prompt').value.trim();
  const primaryText = systemText;
  const contextText = userText;

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
  document.getElementById('statsRow').style.display          = 'none';
  document.getElementById('modelOutputCard').style.display   = 'none';
  document.getElementById('panelActions').style.display      = 'none';
  document.getElementById('thresholdRow').style.display      = 'none';
  document.getElementById('shareBtn').style.display          = 'none';
  document.getElementById('compressResultPanel').style.display = 'none';
  document.getElementById('saliencyText').innerHTML = `
    <div class="empty-state">
      <div class="empty-icon" style="border-color:var(--accent);animation:spin 1.4s linear infinite">◎</div>
      <p>Running saliency analysis…</p>
    </div>`;

  showProgressUI(true);
  setProgress('Segmenting prompt…', 5);

  try {
    const phrases = segmentPrompt(primaryText);
    document.getElementById('phraseCount').textContent =
      `${phrases.length} phrase${phrases.length !== 1 ? 's' : ''}`;

    setProgress(`Getting baseline response (${phrases.length} phrases found)…`, 15);
    const { userMsg: baseUser, systemMsg: baseSys } = buildCall(primaryText);
    const baseline = await callLLM(baseUser, baseSys, 600);

    document.getElementById('modelOutputCard').style.display = 'block';
    document.getElementById('modelResponse').textContent = baseline;

    const isSemantic = selectedSimilarity === SimilarityStrategy.SEMANTIC &&
                       (!!getKey('together') || !!getKey('openai'));
    const onTick = (done, total) => {
      const pct = 22 + Math.round((done / total) * 70);
      setProgress(`Shapley: ${done} / ${total} walks…`, pct);
    };

    const coalitionCall = async (coalitionText) => {
      const { userMsg, systemMsg } = buildCall(coalitionText);
      return callLLM(userMsg, systemMsg, 400);
    };

    const rawScores = await computeShapley(phrases, isSemantic, coalitionCall, onTick);

    setProgress('Normalizing scores and rendering…', 95);

    // Reset threshold and view for fresh run
    scoreThreshold = 0;
    document.getElementById('thresholdSlider').value = '0';
    document.getElementById('thresholdValue').textContent = 'All';

    const normScores = normalise(rawScores);
    renderSaliency(phrases, normScores);
    renderStats(phrases, normScores);
    saveToHistory(phrases, normScores);

    // Sync view-mode buttons then reveal post-analysis controls
    document.getElementById('viewHeatmapBtn').classList.toggle('active', viewMode === 'heatmap');
    document.getElementById('viewTableBtn').classList.toggle('active', viewMode === 'table');
    document.getElementById('panelActions').style.display = '';
    document.getElementById('thresholdRow').style.display = '';
    document.getElementById('shareBtn').style.display     = '';

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

// ─── Compression ──────────────────────────────────────────────────────────────

function countWordsApprox(text) {
  return text.split(/\s+/).filter(Boolean).length;
}

function reconstructPrompt(originalText, diff) {
  let result = originalText;
  for (const entry of diff) {
    if (entry.action === 'keep') continue;
    const orig = entry.original;
    if (result.includes(orig)) {
      result = result.replace(orig, entry.result);
    } else if (result.includes(orig.trim())) {
      // Bullet/markdown phrases are trimmed by segmenter — match without surrounding whitespace
      result = result.replace(orig.trim(), entry.result);
    }
    // If neither matches, skip silently (better than corrupting the prompt)
  }
  result = result.replace(/\n{3,}/g, '\n\n');
  result = result.replace(/\n[ \t]+\n/g, '\n\n');
  result = result.replace(/[ \t]+\n/g, '\n');
  return result.trim();
}

// ── DOM helpers ───────────────────────────────────────────────────────────────

function updateCompressThreshold(val) {
  compressThreshold = parseInt(val, 10) / 100;
  document.getElementById('compressThresholdLabel').textContent = `< ${compressThreshold.toFixed(2)}`;
}

function setCompressProgress(msg, pct) {
  document.getElementById('compressPhaseLog').textContent    = `▶ ${msg}`;
  document.getElementById('compressProgressBar').style.width = `${pct}%`;
}

function showCompressProgress(show) {
  const wrap = document.getElementById('compressProgressWrap');
  const log  = document.getElementById('compressPhaseLog');
  wrap.className = show ? 'progress-wrap active' : 'progress-wrap';
  log.className  = show ? 'phase-log active'     : 'phase-log';
  if (!show) { log.textContent = ''; log.style.color = ''; }
}

function showCompressError(msg) {
  const log = document.getElementById('compressPhaseLog');
  log.textContent = `⚠ ${msg}`;
  log.classList.add('active');
  log.style.color = 'var(--warn)';
}


function renderDiffList(diff) {
  const container = document.getElementById('compressDiffList');
  const changed   = diff.filter(e => e.action !== 'keep');
  if (changed.length === 0) {
    container.innerHTML = '<div class="diff-empty">All phrases retained at this threshold — try lowering the threshold to compress more aggressively.</div>';
    return;
  }
  container.innerHTML = changed.map(entry => {
    const isRemove   = entry.action === 'remove';
    const origClass  = isRemove ? 'diff-text-original diff-text-removed' : 'diff-text-original';
    const resultHtml = isRemove ? '' : `
      <span class="diff-arrow-sep">→</span>
      <span class="diff-text-result">${htmlEsc(entry.result)}</span>`;
    return `<div class="diff-row">
      <span class="diff-action diff-action--${entry.action}">${entry.action}</span>
      <span class="${origClass}">${htmlEsc(entry.original)}</span>
      ${resultHtml}
    </div>`;
  }).join('');
}

function renderCompressResult(compressed, diff, originalText) {
  const origWords  = countWordsApprox(originalText);
  const compWords  = countWordsApprox(compressed);
  const savedWords = origWords - compWords;
  const savedPct   = Math.round((savedWords / Math.max(origWords, 1)) * 100);
  document.getElementById('compressStats').textContent =
    savedWords > 0 ? `~${savedPct}% fewer words (${origWords} → ${compWords})` : 'No word reduction';

  document.getElementById('compressedPromptText').value = compressed;

  renderDiffList(diff);

  const changed = diff.filter(e => e.action !== 'keep').length;
  document.getElementById('diffSummary').textContent =
    changed > 0 ? `${changed} phrase${changed !== 1 ? 's' : ''} changed` : 'No phrases changed';

  document.getElementById('compressDiffList').style.display = 'none';
  document.getElementById('diffToggle').textContent = 'Show diff ▾';

  document.getElementById('compressResultPanel').style.display = '';
  showCompressProgress(false);
}

function toggleDiffList() {
  const list = document.getElementById('compressDiffList');
  const btn  = document.getElementById('diffToggle');
  const open = list.style.display === 'none' || list.style.display === '';
  list.style.display = open ? 'block' : 'none';
  btn.textContent    = open ? 'Hide diff ▴' : 'Show diff ▾';
}

function applyCompression() {
  const compressed = document.getElementById('compressedPromptText').value;
  document.getElementById('systemprompt').value = compressed;
  localStorage.setItem(AUTOSAVE_KEYS.system, compressed);
  const btn  = document.getElementById('applyBtn');
  const orig = btn.textContent;
  btn.textContent = 'Applied ✓';
  setTimeout(() => btn.textContent = orig, 1800);
}

function discardCompression() {
  document.getElementById('compressResultPanel').style.display = 'none';
  lastCompressDiff = null;
}

// ── Orchestrators ─────────────────────────────────────────────────────────────

async function runCompress() {
  if (!lastPhrases || !lastNormScores) {
    showError('Run an analysis first before compressing.');
    return;
  }
  lastOriginalText = document.getElementById('systemprompt').value.trim();
  if (!lastOriginalText) {
    showError('No system prompt to compress.');
    return;
  }

  const compressBtn = document.getElementById('compressBtn');
  compressBtn.disabled = true;

  // Step 1 — score-based removal (instant, deterministic)
  const diff = lastPhrases.map((p, i) => {
    const remove = !p.atomic && lastNormScores[i] < compressThreshold;
    return { phraseIdx: i, action: remove ? 'remove' : 'keep', original: p.text, result: remove ? '' : p.text };
  });
  lastCompressDiff = diff;
  const rough = reconstructPrompt(lastOriginalText, diff);

  // Step 2 — LLM structural cleanup only
  document.getElementById('compressResultPanel').style.display = '';
  showCompressProgress(true);
  setCompressProgress('Fixing structure…', 40);

  const sysPrompt = `You are a prompt structure fixer. Phrases have been removed from a prompt. Your ONLY job is to fix structural or formatting issues caused by the removal — e.g. broken JSON (missing keys, dangling commas), orphaned XML tags, numbered lists that skip values, or stray blank lines. Rules: (1) Do NOT add back any removed content. (2) Do NOT rephrase, rewrite, or change any kept content — not a single word. (3) If the prompt is already structurally sound, return it exactly as given. Output only the fixed prompt, no explanation.`;

  let cleaned = rough;
  try {
    setCompressProgress('Fixing structure…', 60);
    cleaned = await callLLM(rough, sysPrompt, 1500);
    setCompressProgress('Done', 100);
    setTimeout(() => showCompressProgress(false), 800);
  } catch (err) {
    showCompressError(`Structure fix failed (${err.message}) — showing raw removal.`);
    setTimeout(() => showCompressProgress(false), 2500);
  }

  renderCompressResult(cleaned, diff, lastOriginalText);
  compressBtn.disabled = false;
}


// ─── Keyboard shortcuts ───────────────────────────────────────────────────────

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runAnalysis();
  if (e.key === 'Escape') closeKeyModal();
});

document.addEventListener('click', e => {
  if (!e.target.closest('#historyWrap')) {
    document.getElementById('historyDropdown')?.classList.remove('open');
  }
});

// ─── Init ─────────────────────────────────────────────────────────────────────

updateKeyButton();
updateSimilaritySelector();
initAutoSave();
tryLoadShareURL();
renderHistoryBtn();
