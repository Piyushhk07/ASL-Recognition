# ASL Vision — Real-Time American Sign Language Recognition

A professional computer vision application for recognising American Sign Language (ASL) hand gestures in real time using MediaPipe hand landmark detection and a machine learning classifier.

---

## Features

- **Live Recognition** — Real-time letter detection via webcam with MediaPipe landmark overlay
- **Data Collection** — Record your own hand gestures to train a personalised model
- **Model Performance** — Live metrics dashboard (accuracy, F1, loss curves, per-class accuracy)
- **Temporal Smoothing** — 7-frame vote window eliminates jitter and false positives
- **99-dimensional features** — Scale- and translation-invariant feature engineering for robust classification

---

## Stack

| Component | Technology |
|---|---|
| Backend | Python 3.10+, Flask 3.0 |
| Hand Tracking | MediaPipe Tasks API 1.2.0 |
| ML Model | scikit-learn (RandomForest + MLP ensemble) |
| Feature Engineering | Custom 99-d invariant feature extractor |
| Frontend | Vanilla JS, Canvas 2D API, Inter font |
| Design System | Custom CSS — Linear/Vercel-inspired minimal UI |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Piyushhk07/ASL-Recognition.git
cd ASL-Recognition
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the baseline model

The trained model file is not checked in (too large for GitHub). Generate it:

```bash
python train_model.py
```

This trains a RandomForest on synthetic ASL hand shapes (~2 minutes).

### 4. Start the server

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Improving Accuracy with Your Real Hand

The baseline model is trained on synthetic data. For best results, record your own hand landmarks:

1. Go to **http://localhost:5000/collect**
2. For each letter: click the letter → press **Capture 60 Samples** → hold the sign for 8 seconds
3. After collecting all (or a subset of) letters, click **Train on Collected Data**
4. The model reloads automatically — switch to Live Recognition to test

> **Tip:** Even collecting 5–6 letters dramatically improves accuracy for those signs.

---

## Project Structure

```
├── app.py                  # Flask backend (routes, API, MediaPipe)
├── features.py             # 99-d feature extraction (shared by train + inference)
├── train_model.py          # Model training (synthetic + real data support)
├── run_check.py            # 71-check diagnostic script
├── requirements.txt
├── model/
│   ├── asl_model.pkl       # Trained model (gitignored — run train_model.py)
│   ├── hand_landmarker.task # MediaPipe binary (gitignored — auto-downloaded)
│   └── metrics.json        # Training metrics
├── data/
│   └── collected/          # Real hand landmarks (gitignored — user-specific)
├── static/
│   ├── css/main.css        # Full design system
│   └── js/
│       ├── app.js          # Nav, toast, API status polling
│       ├── recognition.js  # Camera, landmark drawing, prediction
│       ├── collect.js      # Data collection flow
│       └── performance.js  # Metrics charts (Canvas 2D)
└── templates/
    ├── base.html           # Sidebar layout
    ├── index.html          # Dashboard
    ├── recognition.html    # Live recognition workspace
    ├── collect.html        # Data collection page
    ├── performance.html    # Model metrics
    └── about.html          # Architecture overview
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Server + model health check |
| `/api/predict` | POST | Run inference on a base64 image |
| `/api/metrics` | GET | Return training metrics JSON |
| `/api/collect` | POST | Save one landmark sample for a letter |
| `/api/collect/status` | GET | Sample counts per letter |
| `/api/collect/clear` | POST | Delete samples for a letter (redo) |
| `/api/train` | POST | Train on collected data (streaming log) |
| `/api/reload_model` | POST | Hot-reload the model file |

---

## Diagnostic

```bash
python run_check.py
```

Runs 71 checks across file integrity, dependencies, model, MediaPipe, HTTP routes, API endpoints, static assets, and HTML content.

---

## Design

Minimal, professional UI inspired by Linear, Vercel, and GitHub. Built without UI frameworks — pure HTML, CSS custom properties, and vanilla JavaScript.

- Font: Inter (Google Fonts)
- Palette: `#F8FAFC` background · `#3157C8` accent · `#172033` text · `#0F1117` video surface
- No gradients, no animations, no glow effects

