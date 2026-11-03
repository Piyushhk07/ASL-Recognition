/**
 * ASL Vision — performance.js
 * Fetches /api/metrics and draws:
 *   - Loss curve (training vs. validation)
 *   - Per-class accuracy bars
 *   - Confusion matrix heatmap (Canvas 2D, no external library)
 */

/* ===================================================================
   DOM REFS
   =================================================================== */
const canvasLoss  = document.getElementById('chart-loss');
const canvasClass = document.getElementById('chart-class');
const canvasCM    = document.getElementById('chart-cm');
const cmCard      = document.getElementById('cm-card');

const mTestAcc    = document.getElementById('m-test-acc');
const mTrainAcc   = document.getElementById('m-train-acc');
const mValAcc     = document.getElementById('m-val-acc');
const mPrecision  = document.getElementById('m-precision');
const mRecall     = document.getElementById('m-recall');
const mF1         = document.getElementById('m-f1');
const cfgAugment  = document.getElementById('cfg-augment');
const cfgFeatures = document.getElementById('cfg-features');
const cfgClasses  = document.getElementById('cfg-classes');
const cfgDataset  = document.getElementById('cfg-dataset');
const btnRefresh  = document.getElementById('btn-refresh-metrics');
const noticeBar   = document.getElementById('accuracy-notice');

/* ===================================================================
   HELPERS
   =================================================================== */
function pct(v) {
  return (v == null) ? '—' : (v * 100).toFixed(1) + '%';
}
function setMetricCard(el, value) {
  if (!el) return;
  el.textContent = pct(value);
}

/* ===================================================================
   CANVAS UTILS
   =================================================================== */
const COLORS = {
  train:   '#3157C8',
  val:     '#94A3B8',
  grid:    '#F1F5F9',
  axis:    '#CBD5E1',
  label:   '#64748B',
  bar:     '#3157C8',
  barLow:  '#DC2626',
};

function setupCanvas(canvas, cssW, cssH) {
  if (!canvas) return null;
  const dpr  = window.devicePixelRatio || 1;
  const rect  = canvas.getBoundingClientRect();
  const w     = cssW || rect.width  || parseInt(canvas.style.width)  || 600;
  const h     = cssH || rect.height || parseInt(canvas.style.height) || 300;
  canvas.width  = w * dpr;
  canvas.height = h * dpr;
  canvas.style.width  = w + 'px';
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w, h };
}

/* ===================================================================
   LOSS CHART
   =================================================================== */
function drawLossChart(trainingLoss, valLoss) {
  if (!canvasLoss) return;
  const { ctx, w, h } = setupCanvas(canvasLoss);
  if (!ctx) return;

  const pad = { top: 20, right: 20, bottom: 40, left: 48 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top  - pad.bottom;

  const allVals = [...trainingLoss, ...valLoss];
  const maxVal  = Math.max(...allVals, 1.0);
  const epochs  = Math.max(trainingLoss.length, valLoss.length);
  const yTicks  = [0, 0.2, 0.4, 0.6, 0.8, 1.0].filter(v => v <= maxVal + 0.05);
  const xStep   = Math.ceil(epochs / 6);

  function toX(i)   { return pad.left + (i / Math.max(epochs - 1, 1)) * plotW; }
  function toY(val) { return pad.top  + (1 - val / maxVal) * plotH; }

  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth   = 1;
  yTicks.forEach(v => {
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
  });
  for (let i = 0; i < epochs; i += xStep) {
    const x = toX(i);
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + plotH); ctx.stroke();
  }

  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle    = COLORS.label;
  ctx.font         = '11px Inter, system-ui, sans-serif';
  ctx.textAlign    = 'right';
  ctx.textBaseline = 'middle';
  yTicks.forEach(v => { ctx.fillText(v.toFixed(1), pad.left - 6, toY(v)); });

  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Loss', 0, 0);
  ctx.restore();

  ctx.textAlign    = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i < epochs; i += xStep) {
    ctx.fillText(i.toString(), toX(i), pad.top + plotH + 8);
  }
  ctx.fillText('Epoch', pad.left + plotW / 2, pad.top + plotH + 24);

  function drawLine(data, color, dashed) {
    if (!data || data.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.lineJoin    = 'round';
    ctx.beginPath();
    data.forEach((val, i) => {
      const x = toX(i), y = toY(val);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawLine(trainingLoss, COLORS.train, false);
  drawLine(valLoss,      COLORS.val,   true);
}

/* ===================================================================
   PER-CLASS ACCURACY CHART
   =================================================================== */
function drawClassChart(classes, perClassAccuracy) {
  if (!canvasClass) return;
  const { ctx, w, h } = setupCanvas(canvasClass);
  if (!ctx) return;

  const pad   = { top: 16, right: 12, bottom: 36, left: 40 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top  - pad.bottom;
  const n     = classes.length;
  const barW  = plotW / n;
  const gap   = barW * 0.2;

  function toY(val) { return pad.top + (1 - val) * plotH; }

  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth   = 1;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
  });

  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  ctx.fillStyle    = COLORS.label;
  ctx.font         = '10px Inter, system-ui, sans-serif';
  ctx.textAlign    = 'right';
  ctx.textBaseline = 'middle';
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    ctx.fillText(Math.round(v * 100) + '%', pad.left - 4, toY(v));
  });

  classes.forEach((cls, i) => {
    const val = perClassAccuracy && perClassAccuracy[cls] != null ? perClassAccuracy[cls] : 0.0;
    const x   = pad.left + i * barW + gap / 2;
    const bw  = barW - gap;
    const bh  = val * plotH;
    const by  = pad.top + plotH - bh;

    ctx.fillStyle = val < 0.8 ? COLORS.barLow : COLORS.bar;
    ctx.fillRect(x, by, bw, bh);

    ctx.fillStyle    = COLORS.label;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';
    ctx.font         = '9px Inter, system-ui, sans-serif';
    ctx.fillText(cls, x + bw / 2, pad.top + plotH + 5);
  });
}

/* ===================================================================
   CONFUSION MATRIX HEATMAP
   =================================================================== */
function drawConfusionMatrix(cm, classes) {
  if (!cm || !canvasCM || !cmCard) return;
  const n = classes.length;
  if (n === 0 || cm.length !== n) return;

  // Show the card
  cmCard.style.display = '';

  // Cell size — make it fill the card width (max ~600px)
  const CELL  = Math.min(Math.floor(560 / n), 26);
  const LABEL = 22;   // space for axis labels
  const totalW = LABEL + n * CELL + 10;
  const totalH = LABEL + n * CELL + 10;

  const cv = setupCanvas(canvasCM, totalW, totalH);
  if (!cv) return;
  const { ctx, w, h } = cv;

  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);

  // Find max value for colour scaling
  const flat = cm.flat();
  const maxVal = Math.max(...flat, 1);

  // Draw cells
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const val  = cm[r][c];
      const norm = val / maxVal;  // 0..1

      // Diagonal (correct): blue scale; off-diagonal: red scale
      let fillColor;
      if (r === c) {
        // Blue: low = white, high = accent blue
        const intensity = Math.round(norm * 255);
        fillColor = `rgb(${255 - Math.round(intensity * 0.81)},${255 - Math.round(intensity * 0.65)},${255 - Math.round(intensity * 0.22)})`;
      } else if (val > 0) {
        // Red: errors — subtle
        const intensity = Math.round(norm * 180);
        fillColor = `rgb(${220},${255 - intensity},${255 - intensity})`;
      } else {
        fillColor = '#F8FAFC';
      }

      const x = LABEL + c * CELL;
      const y = LABEL + r * CELL;
      ctx.fillStyle = fillColor;
      ctx.fillRect(x, y, CELL - 1, CELL - 1);

      // Draw value if cell is large enough
      if (CELL >= 18 && val > 0) {
        ctx.fillStyle    = norm > 0.55 ? '#fff' : '#1E293B';
        ctx.font         = `${Math.max(7, CELL - 12)}px Inter, system-ui, sans-serif`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(val > 999 ? '999+' : val, x + CELL / 2 - 0.5, y + CELL / 2 - 0.5);
      }
    }
  }

  // Axis labels (column = predicted, row = true)
  ctx.fillStyle    = '#475569';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.font         = `${Math.max(7, CELL - 11)}px Inter, system-ui, sans-serif`;
  classes.forEach((cls, i) => {
    // Column header (top)
    ctx.fillText(cls, LABEL + i * CELL + CELL / 2 - 0.5, LABEL / 2);
    // Row header (left)
    ctx.textAlign = 'right';
    ctx.fillText(cls, LABEL - 4, LABEL + i * CELL + CELL / 2 - 0.5);
    ctx.textAlign = 'center';
  });
}

/* ===================================================================
   FETCH & RENDER
   =================================================================== */
async function loadMetrics() {
  try {
    const res  = await fetch('/api/metrics');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Metric cards
    setMetricCard(mTestAcc,   data.test_accuracy);
    setMetricCard(mTrainAcc,  data.train_accuracy);
    setMetricCard(mValAcc,    data.val_accuracy);
    setMetricCard(mPrecision, data.precision);
    setMetricCard(mRecall,    data.recall);
    setMetricCard(mF1,        data.f1_score);

    // Config table
    if (cfgAugment)  cfgAugment.textContent  = data.augmentation || 'rotation±15°, shift±2.5px, brightness 0.65-1.35, zoom 0.88-1.10';
    if (cfgFeatures) cfgFeatures.textContent = data.num_features ? `${data.num_features} (784 pixels + 56 gradients)` : '840 (784 pixels + 56 gradients)';
    if (cfgClasses)  cfgClasses.textContent  = data.classes ? `${data.classes.length} (${data.classes.join(', ')})` : '24 (A–Y, excluding J & Z)';
    if (cfgDataset && data.dataset_size) {
      const tr = data.dataset_size.train || '—';
      const te = data.dataset_size.test  || '—';
      cfgDataset.textContent = `${tr} train  |  ${te} test`;
    }

    // Source notice
    if (noticeBar) {
      const src = data.training_source || '';
      if (src === 'sign_mnist_augmented') {
        noticeBar.style.display = '';
        noticeBar.className     = 'notice-bar notice-info';
        noticeBar.textContent   = 'Metrics measured on Sign MNIST held-out test set. Real-world webcam accuracy may differ and has not yet been measured.';
      } else if (src === 'real_collected') {
        noticeBar.style.display = '';
        noticeBar.className     = 'notice-bar notice-success';
        noticeBar.textContent   = 'Model trained on real webcam samples collected via the Collect page.';
      } else {
        noticeBar.style.display = 'none';
      }
    }

    // Charts
    drawLossChart(data.training_loss || [], data.val_loss || []);
    drawClassChart(data.classes || [], data.per_class_accuracy || {});

    // Confusion matrix
    if (data.confusion_matrix && data.classes) {
      drawConfusionMatrix(data.confusion_matrix, data.classes);
    } else if (cmCard) {
      cmCard.style.display = 'none';
    }

  } catch (err) {
    console.error('[performance.js] Failed to load metrics:', err.message);
    if (window.showToast) { showToast('Failed to load metrics', 'error'); }
  }
}

/* ===================================================================
   INIT
   =================================================================== */
loadMetrics();

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(loadMetrics, 200);
});

if (btnRefresh) {
  btnRefresh.addEventListener('click', () => {
    loadMetrics();
    if (window.showToast) { showToast('Metrics refreshed', 'success', 2000); }
  });
}
