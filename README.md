# ASL Vision — Real-Time American Sign Language Recognition

A professional web application for real-time ASL hand gesture recognition using MediaPipe, scikit-learn, and Flask.

**Test Accuracy (Sign MNIST): 96.81%** &nbsp;|&nbsp; **F1 Macro: 96.92%** &nbsp;|&nbsp; **24 Classes (A–Y, no J or Z)**

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (uses MNIST dataset folder — takes ~15 min)
python train_model.py

# OR skip training and use the pre-trained model, then start the server
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Root Causes Fixed (Real-World Accuracy Improvements)

### Bug 1 — CRITICAL: CLAHE distribution mismatch
**Problem:** CLAHE contrast enhancement was applied at inference (`app.py`) but NOT during training (Sign MNIST CSV = raw pixels/255). This created a domain shift of ~0.14 mean pixel difference — the model saw completely different pixel distributions at inference vs. training.

**Fix:** Removed CLAHE from inference. Created `preprocess.py` as the single shared preprocessing pipeline for both training and inference. Both now use: `grayscale → INTER_AREA resize to 28×28 → divide by 255`.

### Bug 2 — HIGH: No data augmentation in training
**Problem:** Model trained only on studio-quality MNIST images (consistent lighting, clean backgrounds, centered hand). Webcam images have variable lighting, different backgrounds, and off-center hands.

**Fix:** Added 3× augmentation per training image: random rotation ±15°, translation ±2.5px, brightness 0.65–1.35×, zoom 0.88–1.10×. No horizontal flip (would change ASL handedness). Training set grew from 27,455 → 109,820 samples.

### Bug 3 — HIGH: No confidence threshold
**Problem:** Every prediction was accepted and displayed regardless of model confidence, even when the model was uncertain (0.05 confidence → same as 0.95 confidence).

**Fix:** Added `CONF_THRESHOLD = 0.40` in `recognition.js`. Predictions below 40% confidence show "Low confidence (X%)" and are NOT added to gesture history.

### Bug 4 — MEDIUM: Relative bounding box padding (too tight for distant hands)
**Problem:** Padding was `PAD * x_span` and `PAD * y_span` separately — when a hand is far from the camera the span is small so padding shrinks to almost nothing, cropping off fingertips.

**Fix:** Padding is now `30%` of the **longer** bounding box side (absolute, not relative to each axis separately). This scales correctly for hands at any distance.

### Bug 5 — MEDIUM: No letter cooldown (rapid-fire same letter)
**Problem:** The same letter could be auto-added to gesture history dozens of times per second just from holding the pose.

**Fix:** Added `LETTER_COOLDOWN_MS = 1500` — at least 1.5 seconds must pass between auto-accepted letters.

---

## Files Modified

| File | Change |
|------|--------|
| `preprocess.py` | **NEW** — Shared preprocessing pipeline (no CLAHE). Used by both `train_model.py` and `app.py`. |
| `train_model.py` | Rewritten — uses `preprocess.py`, 3× augmentation, RF150 + MLP(512-256-128) ensemble, confusion matrix saved to metrics.json. |
| `app.py` | Removed CLAHE from `_detect_hand_and_extract()`. Uses `preprocess.preprocess_bgr_crop()`. Absolute bounding box padding. Returns `top3` predictions. Added `/api/debug_frame` endpoint. |
| `static/js/recognition.js` | Added `CONF_THRESHOLD`, `LETTER_COOLDOWN_MS`, diagnostic panel support, `/api/debug_frame` integration. |
| `templates/recognition.html` | Added collapsible Diagnostic Mode panel with 28×28 preview canvas and top-3 chart. |
| `templates/performance.html` | Updated training config table (accurate), added confusion matrix card, accuracy source notice. |
| `static/js/performance.js` | Added confusion matrix heatmap renderer (Canvas 2D). Added `notice-bar` for accuracy source. |
| `static/css/main.css` | Added styles for diagnostic panel, notice-bar, btn-warning. |
| `run_check.py` | Added `preprocess.py` to file integrity checks, added `/api/debug_frame` endpoint tests. |

---

## Model Architecture

**RF + MLP Soft-Voting Ensemble**

```
Input: 840-d feature vector
  = 784 pixel values (28×28 flattened)
  + 56 Sobel gradient values

RandomForestClassifier(n_estimators=150, class_weight='balanced')
  └─ Captures spatial pixel co-occurrence

Pipeline([StandardScaler, MLPClassifier(512→256→128, ReLU, Adam)])
  └─ Learns non-linear representations from scaled features

VotingClassifier(voting='soft')  ←  weighted average of probabilities
```

---

## Preprocessing Pipeline (Shared: train + inference)

```
Webcam frame (BGR)
  │
  ▼
MediaPipe HandLandmarker
  → 21 landmarks → bounding box
  → Absolute padding = 30% of longer bbox side
  → Square crop (aspect preserved)
  │
  ▼
BGR → Grayscale (cv2.COLOR_BGR2GRAY)
  │
  ▼
INTER_AREA resize → 28×28
  │
  ▼
float32 / 255  →  [0, 1]
  │
  ▼
extract_features() → 840-d vector
  │
  ▼
VotingClassifier.predict_proba()
  │
  ▼
Confidence threshold (≥ 40%) + Temporal smoothing (4/7 votes)
  + Cooldown (1.5s between auto-adds)
```

**NO CLAHE anywhere in this pipeline.** MNIST CSV pixels are raw (no CLAHE) so inference must be raw too.

---

## Diagnostic Mode

On the **Live Recognition** page, enable **Diagnostic Mode** to see:

- The exact **28×28 grayscale image** the model receives (upscaled 4× with nearest-neighbour)
- **Top-3 predictions** with confidence bars
- **Latency**, **stability counter**, **confidence threshold**, **FPS**

This lets you verify whether a wrong prediction is due to hand detection, cropping, preprocessing, or the classifier.

---

## Model Performance (Sign MNIST Test Set)

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.81%** |
| Train Accuracy | 99.63% |
| Precision (weighted) | 97.04% |
| Recall (weighted) | 96.81% |
| F1 (weighted) | 96.82% |
| F1 (macro) | **96.92%** |
| Training samples | 109,820 (27,455 × 4 with augmentation) |
| Test samples | 7,172 (never augmented) |

> **Note:** These metrics are measured on the Sign MNIST held-out test set.
> Real-world webcam accuracy depends on lighting, hand distance, and background, and has not been formally measured.
> Use the Diagnostic Mode to evaluate your specific webcam setup.

---

## Retraining

```bash
# Default — with augmentation (recommended, ~15 min)
python train_model.py

# Faster — no augmentation (~4 min, lower accuracy)
python train_model.py --no-augment

# Custom dataset folder
python train_model.py --dataset path/to/folder/

# Fallback synthetic data (for testing only)
python train_model.py --synthetic
```

The script automatically backs up the existing model to `model/asl_model_backup.pkl` before overwriting.

After retraining, hot-reload the model without restarting Flask:

```bash
curl -X POST http://localhost:5000/api/reload_model
```

---

## Dataset

**Sign Language MNIST** — place CSVs in `MNIST dataset/`:

```
MNIST dataset/
  sign_mnist_train.csv   (27,455 rows)
  sign_mnist_test.csv    (7,172 rows)
```

Labels 0–24 map to A–Y (skipping J=9 and Z=25 — both require motion, not static pose).

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Server + model health |
| `/api/predict` | POST | Predict from base64 image. Returns `letter`, `confidence`, `top3`, `landmarks` |
| `/api/debug_frame` | POST | Same as predict + `crop_b64` (28×28 PNG) + `preprocess_info` |
| `/api/metrics` | GET | Training metrics, confusion matrix, per-class accuracy |
| `/api/reload_model` | POST | Hot-reload model without Flask restart |
| `/api/collect` | POST | Save real webcam sample |
| `/api/collect/status` | GET | Collected samples per letter |
| `/api/train` | POST | Train on collected webcam data (streaming log) |

---

## Requirements

```
flask>=2.3
flask-cors
opencv-python>=4.8
numpy>=1.24
mediapipe>=1.0
scikit-learn>=1.3
joblib
pandas
Pillow
requests
```

---

## Known Limitations

1. **Real-world accuracy not formally measured** — requires labeled webcam samples.
2. **J and Z not supported** — both require hand motion (dynamic signs), not handled by static-pose model.
3. **Lighting sensitivity** — works best with good frontal lighting on a plain background.
4. **Model size: 1.6 GB** — the RF component is large. A CNN would be smaller and faster.
5. **Inference latency** — VotingClassifier inference takes ~100–300ms on CPU for 840-d features.
