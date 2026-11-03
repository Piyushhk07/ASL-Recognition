/**
 * ASL Vision — recognition.js
 * Handles camera, frame capture, prediction, landmark drawing,
 * FPS stats, sentence builder, and gesture history.
 */

/* ===================================================================
   CONSTANTS
   =================================================================== */
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],          // thumb
  [0,5],[5,6],[6,7],[7,8],          // index
  [0,9],[9,10],[10,11],[11,12],     // middle
  [0,13],[13,14],[14,15],[15,16],   // ring
  [0,17],[17,18],[18,19],[19,20],   // pinky
  [5,9],[9,13],[13,17],             // palm
];

const CAPTURE_FPS        = 10;   // target prediction rate
const HISTORY_MAX        = 8;    // max chips in history

// Temporal smoothing — require a letter to appear in ≥VOTE_THRESHOLD
// of the last VOTE_WINDOW frames before committing it to the display.
const VOTE_WINDOW    = 7;
const VOTE_THRESHOLD = 4;

/* ===================================================================
   STATE
   =================================================================== */
let videoStream      = null;
let isRunning        = false;
let isProcessing     = false;   // prevent concurrent requests
let animFrameId      = null;
let sentence         = '';
let gestureHistory   = [];
let modelLoaded      = false;

// Temporal smoothing buffer
let predictionBuffer = [];   // rolling window of raw letter predictions

// FPS tracking
let lastFrameTime    = 0;
let frameCount       = 0;
let displayFPS       = 0;
let lastFPSUpdate    = 0;

// Capture throttle
let lastCaptureTime  = 0;
const CAPTURE_INTERVAL = 1000 / CAPTURE_FPS;

/* ===================================================================
   DOM REFS
   =================================================================== */
const video          = document.getElementById('video');
const canvasOverlay  = document.getElementById('canvas-overlay');
const ctx            = canvasOverlay ? canvasOverlay.getContext('2d') : null;

const btnStart       = document.getElementById('btn-start');
const btnStop        = document.getElementById('btn-stop');

const videoIdle      = document.getElementById('video-idle');
const videoError     = document.getElementById('video-error');
const videoErrorMsg  = document.getElementById('video-error-msg');

const cameraDot      = document.getElementById('camera-dot');

const statFPS        = document.getElementById('stat-fps');
const statHand       = document.getElementById('stat-hand');
const statLandmarks  = document.getElementById('stat-landmarks');
const statModel      = document.getElementById('stat-model');

const predLetter     = document.getElementById('pred-letter');
const predBadge      = document.getElementById('pred-status-badge');
const confValue      = document.getElementById('conf-value');
const confBar        = document.getElementById('conf-bar');

const sentenceArea   = document.getElementById('sentence-area');
const btnSpace       = document.getElementById('btn-space');
const btnBackspace   = document.getElementById('btn-backspace');
const btnClear       = document.getElementById('btn-clear');
const btnCopy        = document.getElementById('btn-copy');

const historyChips   = document.getElementById('history-chips');
const btnClearHist   = document.getElementById('btn-clear-history');

/* ===================================================================
   CAMERA
   =================================================================== */
async function startCamera() {
  try {
    videoStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = videoStream;
    await video.play();

    isRunning = true;
    hideOverlays();
    updateCameraUI(true);
    lastFrameTime = performance.now();
    lastFPSUpdate = performance.now();
    frameCount    = 0;
    loop();
  } catch (err) {
    showCameraError(err.name === 'NotAllowedError'
      ? 'Camera permission denied. Please allow camera access and try again.'
      : `Camera error: ${err.message}`);
  }
}

function stopCamera() {
  isRunning = false;
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  if (videoStream) { videoStream.getTracks().forEach(t => t.stop()); videoStream = null; }
  video.srcObject = null;
  clearCanvas();
  updateCameraUI(false);
  showIdleOverlay();
  resetStats();
  predictionBuffer = [];   // clear smoothing buffer
  updatePrediction(null, 0);
}

function updateCameraUI(active) {
  if (btnStart) { btnStart.style.display = active ? 'none' : ''; }
  if (btnStop)  { btnStop.style.display  = active ? ''     : 'none'; }
  if (cameraDot) { cameraDot.className = active ? 'camera-status-dot active' : 'camera-status-dot'; }
}

function hideOverlays() {
  if (videoIdle)  { videoIdle.style.display  = 'none'; }
  if (videoError) { videoError.style.display = 'none'; }
}
function showIdleOverlay() {
  if (videoIdle)  { videoIdle.style.display  = ''; }
  if (videoError) { videoError.style.display = 'none'; }
}
function showCameraError(msg) {
  if (videoIdle)  { videoIdle.style.display  = 'none'; }
  if (videoError) { videoError.style.display = ''; }
  if (videoErrorMsg) { videoErrorMsg.textContent = msg; }
}

/* ===================================================================
   MAIN LOOP
   =================================================================== */
function loop(timestamp = 0) {
  if (!isRunning) return;
  animFrameId = requestAnimationFrame(loop);

  // FPS calculation
  const now = performance.now();
  frameCount++;
  if (now - lastFPSUpdate >= 1000) {
    displayFPS = Math.round(frameCount * 1000 / (now - lastFPSUpdate));
    frameCount = 0;
    lastFPSUpdate = now;
    if (statFPS) { statFPS.textContent = displayFPS; }
  }

  // Throttled capture
  if (!isProcessing && (now - lastCaptureTime) >= CAPTURE_INTERVAL) {
    lastCaptureTime = now;
    captureAndPredict();
  }
}

/* ===================================================================
   CAPTURE & PREDICT
   =================================================================== */
function captureFrame() {
  if (!video.videoWidth) return null;
  const tmp = document.createElement('canvas');
  tmp.width  = video.videoWidth;
  tmp.height = video.videoHeight;
  tmp.getContext('2d').drawImage(video, 0, 0);
  return tmp.toDataURL('image/jpeg', 0.8);
}

/**
 * Push the latest raw prediction into the rolling window and return
 * the smoothed letter if it wins a majority vote, or null otherwise.
 */
function getSmoothedPrediction(rawLetter) {
  if (rawLetter) predictionBuffer.push(rawLetter);
  if (predictionBuffer.length > VOTE_WINDOW) predictionBuffer.shift();

  // Tally votes
  const counts = {};
  predictionBuffer.forEach(l => { counts[l] = (counts[l] || 0) + 1; });
  const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return (best && best[1] >= VOTE_THRESHOLD) ? best[0] : null;
}

async function captureAndPredict() {
  if (!isRunning || isProcessing) return;
  const dataUrl = captureFrame();
  if (!dataUrl) return;

  isProcessing = true;
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const handDetected = data.hand_detected === true;
    updateStats(handDetected, data.landmarks ? data.landmarks.length : 0);

    if (data.landmarks && data.landmarks.length > 0) {
      drawLandmarks(data.landmarks);
    } else {
      clearCanvas();
    }

    if (data.letter) {
      // Apply temporal smoothing — only update display when vote is decisive
      const smoothed = getSmoothedPrediction(data.letter);
      if (smoothed) {
        updatePrediction(smoothed, data.confidence || 0);
        addToHistory(smoothed);
      } else {
        // Buffer not settled yet — show raw letter dimmed
        updatePrediction(data.letter, data.confidence || 0, 'Stabilising…');
      }
    } else {
      predictionBuffer = [];   // reset buffer on no-hand frames
      if (!handDetected) {
        updatePrediction(null, 0, 'No hand detected');
      } else if (data.error && data.error.includes('Model')) {
        updatePrediction(null, 0, 'Model not loaded');
      } else {
        updatePrediction(null, 0, 'Detecting…');
      }
    }
  } catch (err) {
    console.error('[recognition.js] Predict error:', err.message);
  } finally {
    isProcessing = false;
  }
}

/* ===================================================================
   LANDMARK DRAWING
   =================================================================== */
function clearCanvas() {
  if (!ctx || !canvasOverlay) return;
  ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
}

function drawLandmarks(landmarks) {
  if (!ctx || !canvasOverlay || !video.videoWidth) return;

  // Match canvas to video display size
  const rect = video.getBoundingClientRect();
  if (canvasOverlay.width !== rect.width || canvasOverlay.height !== rect.height) {
    canvasOverlay.width  = rect.width  || video.videoWidth;
    canvasOverlay.height = rect.height || video.videoHeight;
  }

  const w = canvasOverlay.width;
  const h = canvasOverlay.height;

  ctx.clearRect(0, 0, w, h);

  // Convert normalised [0,1] coords to canvas pixels
  const pts = landmarks.map(([x, y]) => [
    (1 - x) * w,   // mirror horizontally so it feels natural
    y * h,
  ]);

  // Draw connections
  ctx.strokeStyle = 'rgba(52, 211, 153, 0.7)';
  ctx.lineWidth   = 1.8;
  ctx.lineCap     = 'round';
  HAND_CONNECTIONS.forEach(([a, b]) => {
    if (!pts[a] || !pts[b]) return;
    ctx.beginPath();
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
    ctx.stroke();
  });

  // Draw landmark dots
  pts.forEach(([px, py], i) => {
    // Fingertips slightly larger
    const r = [4, 8, 12, 16, 20].includes(i) ? 5 : 3.5;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(52, 211, 153, 0.95)';
    ctx.fill();
    ctx.strokeStyle = '#0F1117';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}

/* ===================================================================
   UI UPDATES
   =================================================================== */
function updatePrediction(letter, confidence, statusOverride = null) {
  if (predLetter) {
    predLetter.textContent = letter || '--';
    predLetter.style.color = letter ? 'var(--text-primary)' : 'var(--text-secondary)';
  }

  const pct = Math.round((confidence || 0) * 100);
  if (confValue) { confValue.textContent = `${pct}%`; }
  if (confBar) {
    confBar.style.width = `${pct}%`;
    confBar.className = 'confidence-fill' + (pct >= 80 ? '' : pct >= 60 ? ' medium' : ' low');
  }

  // Badge
  if (predBadge) {
    let badgeText  = statusOverride || (letter ? 'Detecting…' : 'No hand detected');
    let badgeClass = 'badge';
    if (letter && pct >= 80)      { badgeClass += ' badge-success'; badgeText = `Confident (${pct}%)`; }
    else if (letter)               { badgeClass += ' badge-warning'; badgeText = `Uncertain (${pct}%)`; }
    else if (!isRunning)           { badgeText = 'Inactive'; }
    else if (statusOverride && statusOverride.includes('Model')) { badgeClass += ' badge-error'; }
    predBadge.className  = badgeClass;
    predBadge.textContent = badgeText;
  }
}

function updateStats(handDetected, landmarkCount) {
  if (statHand) {
    statHand.textContent = handDetected ? 'Yes' : 'No';
    statHand.style.color = handDetected ? 'var(--success)' : 'var(--text-secondary)';
  }
  if (statLandmarks) {
    statLandmarks.textContent = handDetected ? landmarkCount.toString() : '--';
  }
  if (statModel) {
    statModel.textContent = modelLoaded ? 'Loaded' : 'Not Loaded';
    statModel.style.color = modelLoaded ? 'var(--success)' : 'var(--warning)';
  }
}

function resetStats() {
  if (statFPS)       { statFPS.textContent       = '--'; }
  if (statHand)      { statHand.textContent      = '--'; statHand.style.color = ''; }
  if (statLandmarks) { statLandmarks.textContent = '--'; }
  if (statModel)     { statModel.textContent     = '--'; statModel.style.color = ''; }
}

/* ===================================================================
   GESTURE HISTORY
   =================================================================== */
function addToHistory(letter) {
  gestureHistory.push(letter);
  if (gestureHistory.length > HISTORY_MAX) {
    gestureHistory.shift();
  }
  renderHistory();
}

function renderHistory() {
  if (!historyChips) return;
  if (gestureHistory.length === 0) {
    historyChips.innerHTML = '<span class="history-empty">No gestures yet</span>';
    return;
  }
  historyChips.innerHTML = '';
  gestureHistory.forEach((l, i) => {
    const chip = document.createElement('div');
    chip.className = `history-chip${i === gestureHistory.length - 1 ? ' latest' : ''}`;
    chip.textContent = l;
    historyChips.appendChild(chip);
  });
}

/* ===================================================================
   SENTENCE BUILDER
   =================================================================== */
function updateSentenceArea() {
  if (sentenceArea) { sentenceArea.value = sentence; }
}

if (btnSpace) {
  btnSpace.addEventListener('click', () => {
    sentence += ' ';
    updateSentenceArea();
  });
}
if (btnBackspace) {
  btnBackspace.addEventListener('click', () => {
    sentence = sentence.slice(0, -1);
    updateSentenceArea();
  });
}
if (btnClear) {
  btnClear.addEventListener('click', () => {
    sentence = '';
    updateSentenceArea();
  });
}
if (btnCopy) {
  btnCopy.addEventListener('click', () => {
    if (!sentence) { showToast('Nothing to copy', 'warning'); return; }
    navigator.clipboard.writeText(sentence)
      .then(() => showToast('Copied to clipboard', 'success'))
      .catch(() => showToast('Copy failed', 'error'));
  });
}

/* Add predicted letter to sentence when user double-clicks letter display */
if (predLetter) {
  predLetter.title = 'Double-click to add to sentence';
  predLetter.style.cursor = 'pointer';
  predLetter.addEventListener('dblclick', () => {
    const l = predLetter.textContent.trim();
    if (l && l !== '--') {
      sentence += l;
      updateSentenceArea();
      showToast(`Added "${l}" to sentence`, 'success', 1500);
    }
  });
}

/* Clear history */
if (btnClearHist) {
  btnClearHist.addEventListener('click', () => {
    gestureHistory = [];
    renderHistory();
  });
}

/* ===================================================================
   BUTTON EVENTS
   =================================================================== */
if (btnStart) { btnStart.addEventListener('click', startCamera); }
if (btnStop)  { btnStop.addEventListener('click',  stopCamera);  }

/* ===================================================================
   API STATUS LISTENER (from app.js)
   =================================================================== */
document.addEventListener('api:status', e => {
  modelLoaded = e.detail.model_loaded === true;
  if (statModel) {
    statModel.textContent = isRunning ? (modelLoaded ? 'Loaded' : 'Not Loaded') : '--';
    statModel.style.color = isRunning && modelLoaded ? 'var(--success)' : (isRunning ? 'var(--warning)' : '');
  }
});

/* ===================================================================
   INIT
   =================================================================== */
showIdleOverlay();
resetStats();

