# -*- coding: utf-8 -*-
"""
ASL Vision -- Full Diagnostic Check
Run: python run_check.py
"""
import sys, os, json, time, base64
import numpy as np
import cv2
import requests

BASE = 'http://localhost:5000'

results = []

def check(label, ok, detail=''):
    icon  = 'PASS' if ok else 'FAIL'
    badge = '  (' + detail + ')' if detail else ''
    print('  [' + icon + ']  ' + label + badge)
    results.append((label, ok))

def section(title):
    print('')
    print('--- ' + title + ' ---')


# --------------------------------------------------------------------------
# 1. FILE INTEGRITY
# --------------------------------------------------------------------------
section('1. FILE INTEGRITY')
files = {
    'app.py':                       5_000,
    'train_model.py':               5_000,
    'features.py':                  2_000,
    'requirements.txt':             50,
    'model/asl_model.pkl':          1_000_000,
    'model/hand_landmarker.task':   1_000_000,
    'model/metrics.json':           200,
    'static/css/main.css':          5_000,
    'static/js/app.js':             500,
    'static/js/recognition.js':     5_000,
    'static/js/performance.js':     3_000,
    'templates/base.html':          1_000,
    'templates/index.html':         1_000,
    'templates/recognition.html':   1_000,
    'templates/performance.html':   1_000,
    'templates/about.html':         1_000,
}
for f, min_bytes in files.items():
    exists = os.path.exists(f)
    size   = os.path.getsize(f) if exists else 0
    ok     = exists and size >= min_bytes
    check(f, ok, '{:,} bytes'.format(size))


# --------------------------------------------------------------------------
# 2. PYTHON DEPENDENCIES
# --------------------------------------------------------------------------
section('2. PYTHON DEPENDENCIES')
deps = [
    ('flask',      'Flask'),
    ('flask_cors', 'Flask-CORS'),
    ('cv2',        'OpenCV'),
    ('numpy',      'NumPy'),
    ('mediapipe',  'MediaPipe'),
    ('sklearn',    'scikit-learn'),
    ('joblib',     'joblib'),
    ('PIL',        'Pillow'),
]
for mod, name in deps:
    try:
        import importlib.metadata as im
        try:
            ver = im.version(mod if mod != 'cv2' else 'opencv-python')
        except Exception:
            m   = __import__(mod)
            ver = getattr(m, '__version__', '?')
        check(name, True, 'v' + ver)
    except Exception as e:
        check(name, False, str(e))


# --------------------------------------------------------------------------
# 3. ML MODEL
# --------------------------------------------------------------------------
section('3. ML MODEL')
try:
    import joblib as jl
    payload = jl.load('model/asl_model.pkl')
    # Support both old (bare clf) and new (dict) formats
    if isinstance(payload, dict):
        model_clf = payload['model']
        le        = payload.get('label_encoder', None)
        check('Model loads (dict format)', True, type(model_clf).__name__)
    else:
        model_clf = payload
        le        = None
        check('Model loads (bare clf)', True, type(model_clf).__name__)

    n_feat = getattr(model_clf, 'n_features_in_', 0)
    n_cls  = getattr(model_clf, 'n_classes_', 0)
    # Accept 63 (old) or 99 (new) feature count
    feat_ok = n_feat in (63, 99)
    check('Feature count is 63 or 99', feat_ok, str(n_feat))
    check('Class count   == 26', n_cls  == 26, str(n_cls))
    dummy = np.random.default_rng(0).random((1, n_feat)).astype(np.float32)
    proba = model_clf.predict_proba(dummy)[0]
    check('predict_proba returns 26 values', len(proba) == 26,
          'sum={:.4f}'.format(proba.sum()))
    if le is not None:
        classes = le.classes_.tolist()
        check('LabelEncoder has 26 classes', len(classes) == 26,
              str(len(classes)))
except Exception as e:
    check('ML model', False, str(e))


# --------------------------------------------------------------------------
# 4. MEDIAPIPE PIPELINE
# --------------------------------------------------------------------------
section('4. MEDIAPIPE PIPELINE')
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base

    TASK_FILE = 'model/hand_landmarker.task'
    task_size = os.path.getsize(TASK_FILE) if os.path.exists(TASK_FILE) else 0
    check('hand_landmarker.task present', task_size > 1_000_000,
          '{:,} bytes'.format(task_size))

    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=TASK_FILE),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )
    det = mp_vision.HandLandmarker.create_from_options(opts)
    check('HandLandmarker initialises', True, 'MediaPipe v' + mp.__version__)

    # blank image -> no hand expected
    blank  = np.zeros((480, 640, 3), np.uint8)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)
    result = det.detect(mp_img)
    n_hands = len(result.hand_landmarks)
    check('Blank image -> 0 hands detected', n_hands == 0,
          str(n_hands) + ' hands')
    det.close()
except Exception as e:
    check('MediaPipe init', False, str(e))


# --------------------------------------------------------------------------
# 5. HTTP ROUTES
# --------------------------------------------------------------------------
section('5. HTTP ROUTES')
routes = ['/', '/recognition', '/performance', '/about']
for route in routes:
    try:
        r  = requests.get(BASE + route, timeout=6)
        ok = r.status_code == 200
        check('GET ' + route, ok,
              'HTTP ' + str(r.status_code) + ', ' + '{:,}'.format(len(r.text)) + ' chars')
    except Exception as e:
        check('GET ' + route, False, str(e))


# --------------------------------------------------------------------------
# 6. API ENDPOINTS
# --------------------------------------------------------------------------
section('6. API ENDPOINTS')

# /api/status
try:
    r = requests.get(BASE + '/api/status', timeout=6)
    d = r.json()
    check('/api/status returns 200',     r.status_code == 200)
    check('model_loaded == true',        d.get('model_loaded') is True,
          str(d.get('model_loaded')))
    check('mediapipe   == true',         d.get('mediapipe') is True,
          str(d.get('mediapipe')))
    check('status field == "running"',   d.get('status') == 'running',
          str(d.get('status')))
except Exception as e:
    check('/api/status', False, str(e))

# /api/metrics
try:
    r = requests.get(BASE + '/api/metrics', timeout=6)
    d = r.json()
    check('/api/metrics returns 200', r.status_code == 200)
    check('Has test_accuracy field',
          isinstance(d.get('test_accuracy'), (int, float)),
          str(d.get('test_accuracy')))
    pca = d.get('per_class_accuracy', {})
    check('per_class_accuracy has 26 entries',
          isinstance(pca, dict) and len(pca) == 26,
          str(len(pca)) + ' entries')
    tl = d.get('training_loss', [])
    check('training_loss has 30 data points',
          isinstance(tl, list) and len(tl) == 30,
          str(len(tl)) + ' points')
    vl = d.get('val_loss', [])
    check('val_loss has 30 data points',
          isinstance(vl, list) and len(vl) == 30,
          str(len(vl)) + ' points')
except Exception as e:
    check('/api/metrics', False, str(e))

# /api/predict -- blank frame, no hand
try:
    img = np.zeros((480, 640, 3), np.uint8)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = 'data:image/jpeg;base64,' + base64.b64encode(buf).decode()
    t0  = time.perf_counter()
    r   = requests.post(BASE + '/api/predict', json={'image': b64}, timeout=12)
    ms  = (time.perf_counter() - t0) * 1000
    d   = r.json()
    check('/api/predict returns 200',            r.status_code == 200)
    check('Blank frame -> hand_detected = false', d.get('hand_detected') is False,
          str(d.get('hand_detected')))
    check('Blank frame -> letter = null',         d.get('letter') is None,
          str(d.get('letter')))
    check('Round-trip latency < 3000 ms',         ms < 3000,
          '{:.0f} ms'.format(ms))
except Exception as e:
    check('/api/predict (blank)', False, str(e))

# /api/predict -- empty body -> 400
try:
    r = requests.post(BASE + '/api/predict', json={}, timeout=6)
    check('/api/predict empty body returns 400',
          r.status_code == 400, 'HTTP ' + str(r.status_code))
except Exception as e:
    check('/api/predict (bad request)', False, str(e))


# --------------------------------------------------------------------------
# 7. STATIC ASSETS
# --------------------------------------------------------------------------
section('7. STATIC ASSETS')
statics = {
    '/static/css/main.css':       (':root',          5_000),
    '/static/js/app.js':          ('checkApiStatus', 500),
    '/static/js/recognition.js':  ('startCamera',    5_000),
    '/static/js/performance.js':  ('drawLossChart',  3_000),
}
for path, (needle, min_len) in statics.items():
    try:
        r  = requests.get(BASE + path, timeout=6)
        found = needle in r.text
        ok = r.status_code == 200 and len(r.text) >= min_len and found
        check(path, ok,
              '{:,} chars, "{}" {}'.format(
                  len(r.text), needle, 'found' if found else 'MISSING'))
    except Exception as e:
        check(path, False, str(e))


# --------------------------------------------------------------------------
# 8. HTML CONTENT CHECKS
# --------------------------------------------------------------------------
section('8. HTML CONTENT CHECKS')
html_checks = [
    ('/',            'Inter',           'Dashboard: Inter font linked'),
    ('/',            'sidebar',         'Dashboard: sidebar present'),
    ('/',            'stats-grid',      'Dashboard: stats grid present'),
    ('/',            'letter-grid',     'Dashboard: alphabet grid present'),
    ('/',            'quickstart-list', 'Dashboard: quick-start guide present'),
    ('/recognition', 'canvas-overlay',  'Recognition: canvas overlay present'),
    ('/recognition', 'conf-bar',        'Recognition: confidence bar present'),
    ('/recognition', 'sentence-area',   'Recognition: sentence area present'),
    ('/recognition', 'history-chips',   'Recognition: gesture history present'),
    ('/recognition', 'btn-start',       'Recognition: start camera button'),
    ('/recognition', 'tech-stats',      'Recognition: tech stats bar present'),
    ('/performance', 'chart-loss',      'Performance: loss chart canvas'),
    ('/performance', 'chart-class',     'Performance: class accuracy chart'),
    ('/performance', 'metrics-grid',    'Performance: metrics grid'),
    ('/performance', 'info-grid',       'Performance: info tables'),
    ('/about',       'arch-flow',       'About: architecture flow diagram'),
    ('/about',       'stack-table',     'About: technology stack table'),
]
page_cache = {}
for route, needle, label in html_checks:
    if route not in page_cache:
        try:
            page_cache[route] = requests.get(BASE + route, timeout=6).text
        except Exception:
            page_cache[route] = ''
    ok = needle in page_cache.get(route, '')
    check(label, ok)


# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed

print('')
print('=' * 56)
if failed == 0:
    print('  ALL ' + str(total) + ' CHECKS PASSED -- SYSTEM HEALTHY')
else:
    print('  ' + str(failed) + '/' + str(total) + ' CHECKS FAILED')
    print('')
    print('  Failed checks:')
    for label, ok in results:
        if not ok:
            print('    [FAIL]  ' + label)
print('=' * 56)
print('')

sys.exit(0 if failed == 0 else 1)
