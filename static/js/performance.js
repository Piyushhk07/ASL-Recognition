/**
 * ASL Vision — performance.js
 * Fetches /api/metrics and draws the loss curve + per-class accuracy charts
 * using the native Canvas 2D API.
 */

/* ===================================================================
   DOM REFS
   =================================================================== */
const canvasLoss  = document.getElementById('chart-loss');
const canvasClass = document.getElementById('chart-class');

const mTestAcc   = document.getElementById('m-test-acc');
const mTrainAcc  = document.getElementById('m-train-acc');
const mValAcc    = document.getElementById('m-val-acc');
const mPrecision = document.getElementById('m-precision');
const mRecall    = document.getElementById('m-recall');
const mF1        = document.getElementById('m-f1');
const cfgEpochs  = document.getElementById('cfg-epochs');
const btnRefresh = document.getElementById('btn-refresh-metrics');

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

function setupCanvas(canvas) {
  if (!canvas) return null;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width  = rect.width  * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, w: rect.width, h: rect.height };
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

  const allVals   = [...trainingLoss, ...valLoss];
  const maxVal    = Math.max(...allVals, 1.0);
  const minVal    = 0;
  const epochs    = Math.max(trainingLoss.length, valLoss.length);
  const yTicks    = [0, 0.2, 0.4, 0.6, 0.8, 1.0].filter(v => v <= maxVal + 0.05);
  const xStep     = Math.ceil(epochs / 6);

  function toX(i)   { return pad.left + (i / (epochs - 1)) * plotW; }
  function toY(val) { return pad.top  + (1 - (val - minVal) / (maxVal - minVal)) * plotH; }

  // Background
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);

  // Grid
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

  // Axes
  ctx.strokeStyle = COLORS.axis;
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Y labels
  ctx.fillStyle  = COLORS.label;
  ctx.font       = '11px Inter, system-ui, sans-serif';
  ctx.textAlign  = 'right';
  ctx.textBaseline = 'middle';
  yTicks.forEach(v => {
    ctx.fillText(v.toFixed(1), pad.left - 6, toY(v));
  });

  // Y axis title
  ctx.save();
  ctx.translate(12, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = 'center';
  ctx.fillText('Loss', 0, 0);
  ctx.restore();

  // X labels
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'top';
  for (let i = 0; i < epochs; i += xStep) {
    ctx.fillText(i.toString(), toX(i), pad.top + plotH + 8);
  }
  ctx.fillText('Epoch', pad.left + plotW / 2, pad.top + plotH + 24);

  // Draw line helper
  function drawLine(data, color, dashed) {
    if (!data || data.length === 0) return;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.setLineDash(dashed ? [4, 3] : []);
    ctx.lineJoin    = 'round';
    ctx.beginPath();
    data.forEach((val, i) => {
      const x = toX(i);
      const y = toY(val);
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

  // Background
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, w, h);

  // Y grid
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth   = 1;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + plotW, y); ctx.stroke();
  });

  // Axes
  ctx.strokeStyle = COLORS.axis;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Y labels
  ctx.fillStyle    = COLORS.label;
  ctx.font         = '10px Inter, system-ui, sans-serif';
  ctx.textAlign    = 'right';
  ctx.textBaseline = 'middle';
  [0, 0.25, 0.5, 0.75, 1.0].forEach(v => {
    ctx.fillText(Math.round(v * 100) + '%', pad.left - 4, toY(v));
  });

  // Bars
  classes.forEach((cls, i) => {
    const val  = perClassAccuracy && perClassAccuracy[cls] != null ? perClassAccuracy[cls] : 0.0;
    const x    = pad.left + i * barW + gap / 2;
    const bw   = barW - gap;
    const bh   = val * plotH;
    const by   = pad.top + plotH - bh;

    ctx.fillStyle = val < 0.8 ? COLORS.barLow : COLORS.bar;
    ctx.fillRect(x, by, bw, bh);

    // Letter label
    ctx.fillStyle    = COLORS.label;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';
    ctx.font         = '9px Inter, system-ui, sans-serif';
    ctx.fillText(cls, x + bw / 2, pad.top + plotH + 5);
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

    setMetricCard(mTestAcc,   data.test_accuracy);
    setMetricCard(mTrainAcc,  data.train_accuracy);
    setMetricCard(mValAcc,    data.val_accuracy);
    setMetricCard(mPrecision, data.precision);
    setMetricCard(mRecall,    data.recall);
    setMetricCard(mF1,        data.f1_score);

    if (cfgEpochs) { cfgEpochs.textContent = data.epochs || '—'; }

    // Charts
    drawLossChart(data.training_loss || [], data.val_loss || []);
    drawClassChart(data.classes || [], data.per_class_accuracy || {});
  } catch (err) {
    console.error('[performance.js] Failed to load metrics:', err.message);
    if (window.showToast) { showToast('Failed to load metrics', 'error'); }
  }
}

/* ===================================================================
   INIT
   =================================================================== */
loadMetrics();

// Redraw on resize to keep canvas sharp
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

