/**
 * collect.js — Real-time landmark data collection for ASL Vision
 *
 * Flow:
 *  1. User selects a letter from the grid
 *  2. Presses "Capture 60 Samples"
 *  3. 3-second countdown, then auto-captures frames at ~8fps for ~8s (60 samples)
 *  4. Each frame is POSTed to /api/collect -> server extracts landmarks, saves to disk
 *  5. Progress grid updates; once ≥1 letter is ready, "Train" button enables
 */

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const SAMPLES_PER_LETTER = 60;
const CAPTURE_FPS = 8;

/* ── State ─────────────────────────────────────────────── */
let videoStream     = null;
let isRunning       = false;
let currentLetter   = 'A';
let isCapturing     = false;
let sampleCounts    = {};   // letter -> count on server
let captureInterval = null;

/* ── DOM ─────────────────────────────────────────────────── */
const video             = document.getElementById('video');
const canvasOverlay     = document.getElementById('canvas-overlay');
const ctx               = canvasOverlay ? canvasOverlay.getContext('2d') : null;
const videoIdle         = document.getElementById('video-idle');
const cameraDot         = document.getElementById('camera-dot');
const cameraLabel       = document.getElementById('camera-label');
const btnStart          = document.getElementById('btn-start');
const btnStop           = document.getElementById('btn-stop');
const btnCapture        = document.getElementById('btn-capture');
const btnRedo           = document.getElementById('btn-redo');
const btnTrain          = document.getElementById('btn-train');
const currentLetterEl   = document.getElementById('current-letter');
const letterStatusMsg   = document.getElementById('letter-status-msg');
const letterSampleCount = document.getElementById('letter-sample-count');
const countdownOverlay  = document.getElementById('countdown-overlay');
const countdownNum      = document.getElementById('countdown-num');
const countdownLabel    = document.getElementById('countdown-label');
const recordingOverlay  = document.getElementById('recording-overlay');
const recCount          = document.getElementById('rec-count');
const letterSelectGrid  = document.getElementById('letter-select-grid');
const letterProgGrid    = document.getElementById('letter-progress-grid');
const overviewCount     = document.getElementById('overview-count');
const overviewBar       = document.getElementById('overview-bar');
const trainLogSection   = document.getElementById('train-log-section');
const trainLog          = document.getElementById('train-log');
const trainStatusBadge  = document.getElementById('train-status-badge');

/* ── Build grids ─────────────────────────────────────────── */
function buildGrids() {
  // Letter selector (9 cols, 26 cells + extra for spacing)
  letterSelectGrid.innerHTML = '';
  LETTERS.forEach(l => {
    const cell = document.createElement('div');
    cell.className = 'ls-cell' + (l === currentLetter ? ' active' : '');
    cell.textContent = l;
    cell.dataset.letter = l;
    cell.addEventListener('click', () => selectLetter(l));
    letterSelectGrid.appendChild(cell);
  });

  // Progress strip (2 rows of 13)
  letterProgGrid.innerHTML = '';
  LETTERS.forEach(l => {
    const cell = document.createElement('div');
    cell.className = 'lp-cell';
    cell.id = `lp-${l}`;
    cell.title = l;
    letterProgGrid.appendChild(cell);
  });
}

function selectLetter(l) {
  if (isCapturing) return;
  currentLetter = l;
  currentLetterEl.textContent = l;

  // Update active state in selector
  document.querySelectorAll('.ls-cell').forEach(c => {
    c.classList.toggle('active', c.dataset.letter === l);
  });

  const count = sampleCounts[l] || 0;
  letterSampleCount.textContent = `${count} samples`;
  letterStatusMsg.textContent = count >= SAMPLES_PER_LETTER
    ? `Ready — ${count} samples collected`
    : count > 0
      ? `${count} samples — need ${SAMPLES_PER_LETTER - count} more`
      : 'Hold the sign steady, then press Capture';
  btnRedo.disabled = count === 0 || !isRunning;
  btnCapture.disabled = !isRunning;
}

/* ── Camera ──────────────────────────────────────────────── */
async function startCamera() {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = videoStream;
    await video.play();
    isRunning = true;
    if (videoIdle) videoIdle.style.display = 'none';
    cameraDot.className = 'camera-status-dot active';
    cameraLabel.textContent = 'Live';
    btnStart.style.display = 'none';
    btnStop.style.display = '';
    btnCapture.disabled = false;
    selectLetter(currentLetter);
  } catch (err) {
    alert('Camera error: ' + err.message);
  }
}

function stopCamera() {
  if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
  isRunning = false;
  isCapturing = false;
  if (videoStream) { videoStream.getTracks().forEach(t => t.stop()); videoStream = null; }
  video.srcObject = null;
  cameraDot.className = 'camera-status-dot';
  cameraLabel.textContent = 'Inactive';
  btnStart.style.display = '';
  btnStop.style.display = 'none';
  btnCapture.disabled = true;
  btnRedo.disabled = true;
  hideOverlays();
}

function hideOverlays() {
  countdownOverlay.style.display = 'none';
  recordingOverlay.style.display = 'none';
}

/* ── Capture flow ────────────────────────────────────────── */
async function startCapture() {
  if (!isRunning || isCapturing) return;
  isCapturing = true;
  btnCapture.disabled = true;
  btnRedo.disabled = true;

  // 3-second countdown
  for (let i = 3; i >= 1; i--) {
    countdownNum.textContent = i;
    countdownLabel.textContent = `Sign "${currentLetter}" and hold steady`;
    countdownOverlay.style.display = 'flex';
    await sleep(1000);
  }
  countdownOverlay.style.display = 'none';

  // Record
  recordingOverlay.style.display = 'flex';
  let collected = 0;
  const target = SAMPLES_PER_LETTER;

  await new Promise(resolve => {
    captureInterval = setInterval(async () => {
      if (!isRunning || collected >= target) {
        clearInterval(captureInterval);
        captureInterval = null;
        resolve();
        return;
      }
      const frame = captureFrame();
      if (!frame) return;
      try {
        const r = await fetch('/api/collect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: frame, letter: currentLetter }),
        });
        const d = await r.json();
        if (d.saved) {
          collected++;
          sampleCounts[currentLetter] = (sampleCounts[currentLetter] || 0) + 1;
          recCount.textContent = `${collected} / ${target}`;
          updateProgressUI();
        }
      } catch (_) {}
    }, Math.round(1000 / CAPTURE_FPS));
  });

  hideOverlays();
  isCapturing = false;
  btnCapture.disabled = false;

  selectLetter(currentLetter);   // refresh count display
  // Auto-advance to next uncollected letter
  autoAdvance();
}

function autoAdvance() {
  const next = LETTERS.find(l => (sampleCounts[l] || 0) < SAMPLES_PER_LETTER);
  if (next && next !== currentLetter) {
    selectLetter(next);
    letterStatusMsg.textContent = `Auto-advanced to "${next}" — press Capture when ready`;
  }
}

async function redoLetter() {
  if (!isRunning || isCapturing) return;
  try {
    await fetch('/api/collect/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ letter: currentLetter }),
    });
    sampleCounts[currentLetter] = 0;
    updateProgressUI();
    selectLetter(currentLetter);
  } catch (e) {
    console.error(e);
  }
}

/* ── Helpers ─────────────────────────────────────────────── */
function captureFrame() {
  if (!video.videoWidth) return null;
  const tmp = document.createElement('canvas');
  tmp.width  = video.videoWidth;
  tmp.height = video.videoHeight;
  tmp.getContext('2d').drawImage(video, 0, 0);
  return tmp.toDataURL('image/jpeg', 0.85);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function updateProgressUI() {
  let ready = 0;
  LETTERS.forEach(l => {
    const count = sampleCounts[l] || 0;
    const cell  = document.getElementById(`lp-${l}`);
    const lsCell = document.querySelector(`.ls-cell[data-letter="${l}"]`);
    if (count >= SAMPLES_PER_LETTER) {
      ready++;
      if (cell) { cell.className = 'lp-cell done'; }
      if (lsCell) { lsCell.classList.add('done'); }
    } else if (count > 0) {
      if (cell) { cell.className = 'lp-cell partial'; }
    } else {
      if (cell) { cell.className = 'lp-cell'; }
    }
  });
  overviewCount.textContent = `${ready} / 26 letters ready`;
  overviewBar.style.width = `${Math.round(ready / 26 * 100)}%`;
  btnTrain.disabled = ready === 0;
}

/* ── Load existing counts from server ────────────────────── */
async function loadServerCounts() {
  try {
    const r = await fetch('/api/collect/status');
    const d = await r.json();
    sampleCounts = d.counts || {};
    updateProgressUI();
    selectLetter(currentLetter);
  } catch (_) {}
}

/* ── Training ────────────────────────────────────────────── */
async function trainModel() {
  btnTrain.disabled = true;
  trainLogSection.style.display = '';
  trainLog.textContent = 'Starting training on collected data...\n';
  trainStatusBadge.textContent = 'Training';
  trainStatusBadge.className = 'badge badge-warning';

  try {
    const r = await fetch('/api/train', { method: 'POST' });
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      trainLog.textContent += dec.decode(value);
      trainLog.scrollTop = trainLog.scrollHeight;
    }
    trainStatusBadge.textContent = 'Complete';
    trainStatusBadge.className = 'badge badge-success';
    trainLog.textContent += '\nModel saved. Reloading status...\n';
    // Trigger server-side model reload
    await fetch('/api/reload_model', { method: 'POST' });
    trainLog.textContent += 'Done! Go to Live Recognition to test your new model.\n';
  } catch (e) {
    trainLog.textContent += '\nError: ' + e.message;
    trainStatusBadge.textContent = 'Error';
    trainStatusBadge.className = 'badge badge-error';
  }
  btnTrain.disabled = false;
}

/* ── Event bindings ──────────────────────────────────────── */
if (btnStart)   btnStart.addEventListener('click', startCamera);
if (btnStop)    btnStop.addEventListener('click', stopCamera);
if (btnCapture) btnCapture.addEventListener('click', startCapture);
if (btnRedo)    btnRedo.addEventListener('click', redoLetter);
if (btnTrain)   btnTrain.addEventListener('click', trainModel);

/* ── Init ────────────────────────────────────────────────── */
buildGrids();
loadServerCounts();

