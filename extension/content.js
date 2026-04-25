// Anti-Deepfake-Box content script
// Finds the largest <video> element and overlays deepfake detection results.

const SERVER = 'http://localhost:8000';
const SESSION_ID = 'ext_' + Math.random().toString(36).slice(2, 9);
let active = false;
let intervalId = null;
let overlay = null;
let currentMode = 'online';   // kept in sync with chrome.storage

// Load saved mode; update whenever the user changes it in the popup
chrome.storage.local.get(['adbMode'], result => {
  currentMode = result.adbMode || 'online';
});
chrome.storage.onChanged.addListener(changes => {
  if (changes.adbMode) currentMode = changes.adbMode.newValue;
});

function getLargestVideo() {
  const videos = Array.from(document.querySelectorAll('video'));
  if (!videos.length) return null;
  return videos.reduce((a, b) =>
    (a.videoWidth * a.videoHeight) >= (b.videoWidth * b.videoHeight) ? a : b
  );
}

function createOverlay(video) {
  const el = document.createElement('div');
  el.id = 'adb-overlay';
  Object.assign(el.style, {
    position: 'absolute', top: '8px', left: '8px', zIndex: 99999,
    background: 'rgba(0,0,0,0.72)', color: '#fff',
    fontFamily: '"Inter", system-ui, monospace', fontSize: '12px',
    padding: '6px 10px', borderRadius: '8px',
    pointerEvents: 'none', transition: 'border 0.3s',
    border: '2px solid #888', minWidth: '150px',
  });
  el.innerHTML = '<b>ADB</b> — waiting...';

  const wrapper = document.createElement('div');
  Object.assign(wrapper.style, { position: 'relative', display: 'inline-block' });
  video.parentNode.insertBefore(wrapper, video);
  wrapper.appendChild(video);
  wrapper.appendChild(el);
  return el;
}

function scoreToColor(score, confidence) {
  if (confidence === 'low') return '#888';
  if (score > 0.65) return '#dc2626';   // red = fake
  if (score > 0.45) return '#d97706';   // amber = uncertain
  return '#16a34a';                      // green = real
}

function updateOverlay(data) {
  if (!overlay) return;
  const s = data.smoothed_score;
  const pct = Math.round(s * 100);
  const color = scoreToColor(s, data.confidence);
  const label = s > 0.5 ? 'FAKE' : 'REAL';
  const threshold = data.threshold != null ? Math.round(data.threshold * 100) : 50;
  const modeIcon = data.mode === 'offline' ? '🔬' : '⚡';
  overlay.style.border = `2px solid ${color}`;
  overlay.innerHTML = `
    <b style="color:${color}">${modeIcon} ${label}</b> ${pct}%<br>
    <span style="color:#aaa;font-size:10px">
      vis:${fmt(data.modality_scores?.visual)}
      rppg:${fmt(data.modality_scores?.rppg)}
      fft:${fmt(data.modality_scores?.fft)}<br>
      ${data.fps}fps · ${data.latency_ms}ms · thr:${threshold}% · ${data.confidence}
    </span>`;
}

function fmt(v) { return v != null ? (v * 100).toFixed(0) + '%' : 'N/A'; }

function captureAndSend(video) {
  if (!video || video.readyState < 2) return;
  const canvas = document.createElement('canvas');
  canvas.width  = Math.min(video.videoWidth,  640);
  canvas.height = Math.min(video.videoHeight, 360);
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  const b64 = canvas.toDataURL('image/jpeg', 0.82).split(',')[1];

  fetch(`${SERVER}/detect/frame`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: b64, session_id: SESSION_ID, mode: currentMode }),
  })
  .then(r => r.json())
  .then(data => updateOverlay(data))
  .catch(() => { if (overlay) overlay.innerHTML = '<b>ADB</b> — server offline'; });
}

function start() {
  const video = getLargestVideo();
  if (!video) { setTimeout(start, 2000); return; }
  if (!overlay) overlay = createOverlay(video);
  active = true;
  intervalId = setInterval(() => captureAndSend(video), 1000);  // 1 FPS
}

function stop() {
  active = false;
  clearInterval(intervalId);
  if (overlay) { overlay.remove(); overlay = null; }
}

// Auto-start when video detected
const obs = new MutationObserver(() => {
  if (!active && getLargestVideo()) start();
});
obs.observe(document.body, { childList: true, subtree: true });
if (getLargestVideo()) start();

// Listen for popup toggle
chrome.runtime.onMessage.addListener(msg => {
  if (msg.action === 'toggle') active ? stop() : start();
});
