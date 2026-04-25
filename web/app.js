// Anti-Deepfake Box — dashboard frontend

const SERVER = '';   // same origin (served by FastAPI)
const POLL_MS = 2000;

// ── Theme ────────────────────────────────────────────────────────────────────

const themeBtn = document.getElementById('theme-btn');
let dark = localStorage.getItem('adb-theme') === 'dark'
  || (!localStorage.getItem('adb-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);

function applyTheme() {
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  themeBtn.textContent = dark ? 'Light' : 'Dark';
}
applyTheme();

themeBtn.addEventListener('click', () => {
  dark = !dark;
  localStorage.setItem('adb-theme', dark ? 'dark' : 'light');
  applyTheme();
});

// ── Mode selection ────────────────────────────────────────────────────────────

let currentMode = localStorage.getItem('adb-mode') || 'online';

const modeCards = document.querySelectorAll('.mode-card');
const modeNameEl = document.getElementById('mode-name');

function selectMode(mode) {
  currentMode = mode;
  localStorage.setItem('adb-mode', mode);
  modeCards.forEach(c => {
    c.setAttribute('data-selected', c.dataset.mode === mode ? 'true' : 'false');
  });
  const card = document.querySelector(`.mode-card[data-mode="${mode}"]`);
  modeNameEl.textContent = card ? card.querySelector('.mode-card-title').textContent : mode;

  // Notify server so extension-sourced frames inherit this default
  fetch(`${SERVER}/mode`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  }).catch(() => {});
}

// Restore saved mode
selectMode(currentMode);

modeCards.forEach(card => {
  card.addEventListener('click', () => selectMode(card.dataset.mode));
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectMode(card.dataset.mode); }
  });
});

// ── Score helpers ─────────────────────────────────────────────────────────────

function scoreColor(score) {
  if (score > 0.65) return 'var(--fake)';
  if (score > 0.45) return 'var(--uncertain)';
  return 'var(--real)';
}

function verdictClass(score) {
  if (score > 0.65) return 'fake';
  if (score > 0.45) return 'unsure';
  return 'real';
}

function verdictLabel(score) {
  if (score > 0.65) return 'FAKE';
  if (score > 0.45) return '~';
  return 'REAL';
}

function pct(v) { return v != null ? Math.round(v * 100) + '%' : null; }
function fmt(v) { return v != null ? Math.round(v * 100) + '%' : 'N/A'; }

// ── Render sessions ───────────────────────────────────────────────────────────

const emptyEl = document.getElementById('live-empty');
const container = document.getElementById('sessions-container');

function renderSessions(sessions) {
  const ids = Object.keys(sessions);
  emptyEl.style.display = ids.length ? 'none' : 'block';

  // Remove cards for closed sessions
  container.querySelectorAll('.session-card').forEach(el => {
    if (!ids.includes(el.dataset.session)) el.remove();
  });

  ids.forEach(sid => {
    const d = sessions[sid];
    let card = container.querySelector(`.session-card[data-session="${sid}"]`);
    const score = d.smoothed_score ?? 0.5;
    const ms = d.modality_scores || {};
    const threshold = d.threshold ?? 0.5;
    const mode = d.mode || currentMode;
    const modeLabel = mode === 'offline' ? 'Forensic' : 'Real-time';

    const modRows = [
      { key: 'visual', label: 'Visual' },
      { key: 'rppg',   label: 'rPPG' },
      { key: 'sync',   label: 'Sync' },
      { key: 'fft',    label: 'FFT' },
    ].map(({ key, label }) => {
      const v = ms[key];
      const isNA = v == null;
      const naReason = key === 'sync' && mode === 'online' ? ' — Real-time mode' : '';
      return `
        <div class="mod-row">
          <span class="mod-name">${label}</span>
          <div class="mod-track">
            ${isNA ? '' : `<div class="mod-fill" style="width:${pct(v)};background:${scoreColor(v)}"></div>`}
          </div>
          ${isNA
            ? `<span class="mod-na">N/A${naReason}</span>`
            : `<span class="mod-val">${pct(v)}</span>`}
        </div>`;
    }).join('');

    const html = `
      <div class="session-header">
        <span class="session-id">${sid}</span>
        <span class="live-dot">LIVE</span>
      </div>
      <div class="session-body">
        <div class="verdict">
          <span class="verdict-badge ${verdictClass(score)}">${verdictLabel(score)}</span>
          <div class="verdict-bar-wrap">
            <div class="verdict-bar-fill" style="width:${pct(score)};background:${scoreColor(score)}"></div>
          </div>
          <div>
            <div class="verdict-pct" style="color:${scoreColor(score)}">${pct(score)}</div>
            <div class="verdict-confidence">${(d.confidence || 'low')} confidence</div>
          </div>
        </div>
        <div class="modalities">${modRows}</div>
      </div>
      <div class="session-footer">
        <span><b>Latency</b> ${d.latency_ms ?? '—'}ms</span>
        <span><b>FPS</b> ${d.fps ?? '—'}</span>
        <span><b>Mode</b> ${modeLabel}</span>
        <span><b>Threshold</b> ${Math.round(threshold * 100)}%</span>
        <span><b>Quality</b> ${d.quality ?? '—'}</span>
      </div>`;

    if (card) {
      card.innerHTML = html;
    } else {
      card = document.createElement('div');
      card.className = 'session-card';
      card.dataset.session = sid;
      card.innerHTML = html;
      container.appendChild(card);
    }
  });
}

// ── Polling ───────────────────────────────────────────────────────────────────

async function pollSessions() {
  try {
    const res = await fetch(`${SERVER}/sessions`);
    if (!res.ok) return;
    const data = await res.json();
    renderSessions(data);
  } catch (_) {}
}

pollSessions();
setInterval(pollSessions, POLL_MS);
