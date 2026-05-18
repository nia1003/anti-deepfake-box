'use strict';

const SERVER = 'http://localhost:8000';
const COOLDOWN_MS = 10_000;

let enabled = false;

// ── Cooldown helper ───────────────────────────────────────────────────────────
// Disables btn for COOLDOWN_MS after fn() is called; shows countdown in label.
function withCooldown(btn, fn) {
  if (btn.disabled) return;
  const originalLabel = btn.textContent;
  btn.disabled = true;
  fn();
  let remaining = COOLDOWN_MS / 1000;
  const tick = setInterval(() => {
    remaining--;
    if (remaining <= 0) {
      clearInterval(tick);
      btn.disabled = false;
      btn.textContent = originalLabel;
    } else {
      btn.textContent = `${originalLabel} (${remaining}s)`;
    }
  }, 1000);
}

// ── Network status ────────────────────────────────────────────────────────────
const netEl = document.getElementById('netStatus');

function updateNetworkStatus() {
  const online = navigator.onLine;
  netEl.textContent = online ? 'online' : 'offline';
  netEl.className = online ? 'net-ok' : 'net-err';
}

window.addEventListener('online',  updateNetworkStatus);
window.addEventListener('offline', updateNetworkStatus);
updateNetworkStatus();

// ── Server health (lazy — triggered once on popup open) ───────────────────────
function checkServer() {
  const el = document.getElementById('srv');
  if (!navigator.onLine) { el.textContent = 'no internet'; return; }
  fetch(`${SERVER}/health`, { signal: AbortSignal.timeout(4000) })
    .then(r => r.ok ? r.json() : Promise.reject())
    .then(() => { el.textContent = 'online ✓'; })
    .catch(() => { el.textContent = 'offline ✗'; });
}
checkServer();

// ── Toggle detection ──────────────────────────────────────────────────────────
const toggleBtn = document.getElementById('toggle');

toggleBtn.addEventListener('click', () => {
  if (!navigator.onLine) {
    netEl.textContent = 'no internet — cannot toggle';
    netEl.className = 'net-err';
    return;
  }
  withCooldown(toggleBtn, () => {
    enabled = !enabled;
    toggleBtn.textContent = enabled ? 'Disable Detection' : 'Enable Detection';
    toggleBtn.classList.toggle('off', enabled);
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      if (tabs[0]) chrome.tabs.sendMessage(tabs[0].id, { action: 'toggle' });
    });
  });
});

// ── Email report button ───────────────────────────────────────────────────────
const reportBtn    = document.getElementById('report');
const reportStatus = document.getElementById('reportStatus');

reportBtn.addEventListener('click', () => {
  if (!navigator.onLine) {
    reportStatus.textContent = 'No internet connection.';
    return;
  }
  withCooldown(reportBtn, () => {
    reportStatus.textContent = 'Sending…';
    fetch(`${SERVER}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: 'popup' }),
      signal: AbortSignal.timeout(8000),
    })
      .then(r => r.json())
      .then(d => {
        reportStatus.textContent = d.sent ? 'Report sent ✓' : (d.detail ?? 'Failed');
      })
      .catch(() => {
        reportStatus.textContent = 'Server offline.';
      });
  });
});
