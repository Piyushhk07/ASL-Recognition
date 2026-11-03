/**
 * ASL Vision — app.js
 * Shared logic: nav highlighting, API status check, toast system.
 */

/* ===== NAV HIGHLIGHTING ===== */
(function initNav() {
  const page = document.body.dataset.page || '';
  const navItems = document.querySelectorAll('.nav-item[data-nav]');
  navItems.forEach(item => {
    if (item.dataset.nav === page) {
      item.classList.add('active');
    }
  });
})();

/* ===== API STATUS ===== */
async function checkApiStatus() {
  const dot  = document.getElementById('sidebar-status-dot');
  const text = document.getElementById('sidebar-status-text');
  const modelText = document.getElementById('sidebar-model-text');

  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (dot)  { dot.className = 'status-dot online'; }
    if (text) { text.textContent = 'Server running'; }

    const modelLoaded = data.model_loaded === true;
    const mpAvail     = data.mediapipe   === true;

    if (modelText) {
      if (modelLoaded) {
        modelText.textContent = 'Model: loaded ✓';
        modelText.style.color = 'var(--success)';
      } else {
        modelText.textContent = 'Model: not loaded';
        modelText.style.color = 'var(--warning)';
      }
    }

    // Broadcast for pages that need it
    document.dispatchEvent(new CustomEvent('api:status', { detail: data }));
  } catch (err) {
    if (dot)  { dot.className = 'status-dot offline'; }
    if (text) { text.textContent = 'Server offline'; }
    if (modelText) {
      modelText.textContent = 'Model: —';
      modelText.style.color = '';
    }
    console.warn('[app.js] Status check failed:', err.message);
  }
}

// Run on load
checkApiStatus();
// Refresh every 30 s
setInterval(checkApiStatus, 30_000);

/* ===== TOAST SYSTEM ===== */
const toastContainer = document.getElementById('toast-container');

/**
 * Show a toast notification.
 * @param {string} message
 * @param {'info'|'success'|'error'|'warning'} [type='info']
 * @param {number} [duration=3000] ms
 */
function showToast(message, type = 'info', duration = 3000) {
  if (!toastContainer) return;

  const el = document.createElement('div');
  el.className = `toast${type !== 'info' ? ' ' + type : ''}`;
  el.textContent = message;
  toastContainer.appendChild(el);

  // Trigger transition
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { el.classList.add('show'); });
  });

  setTimeout(() => {
    el.classList.remove('show');
    el.addEventListener('transitionend', () => el.remove(), { once: true });
  }, duration);
}

// Expose globally
window.showToast = showToast;

