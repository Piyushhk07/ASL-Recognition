"""
preprocess.py -- Shared preprocessing pipeline for ASL Sign MNIST
==================================================================
This module is the single source of truth for image preprocessing.
It is used identically by:
  - train_model.py  (training on MNIST CSV data + augmented webcam crops)
  - app.py          (real-time webcam inference)

Maintaining exact parity between training and inference is the #1 requirement
for good real-world accuracy.

Key decisions:
  - NO CLAHE: not applied during training (MNIST CSV = pre-processed pixels).
              Applying it only at inference creates a distribution mismatch.
  - Consistent resize: INTER_AREA downscale for both train and inference.
  - Normalization: divide by 255 only. No Z-score (model sees [0,1] floats).
  - Augmentation: random rotation / brightness / shift only during TRAINING.
"""

import cv2
import numpy as np

IMG_SIZE = 28   # Sign MNIST image dimensions


# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing (train + inference)
# ─────────────────────────────────────────────────────────────────────────────

def to_28x28(gray_crop):
    """
    Resize a grayscale crop to 28×28 using INTER_AREA.
    Expects a uint8 or float32 numpy array.
    Returns float32 array with values in [0, 1].
    """
    if gray_crop.dtype != np.uint8:
        # Convert float [0,1] → uint8 for consistent resize
        gray_crop = np.clip(gray_crop * 255, 0, 255).astype(np.uint8)
    resized = cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def preprocess_bgr_crop(bgr_crop):
    """
    Full preprocessing for a BGR webcam hand crop:
      1. Convert to grayscale
      2. Resize to 28×28 (INTER_AREA)
      3. Normalize to [0, 1]

    Deliberately matches what the MNIST CSV already represents:
    simple grayscale, no extra contrast enhancement.
    """
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return to_28x28(gray)


def preprocess_mnist_row(pixel_row_784):
    """
    Preprocess one row from the Sign MNIST CSV.
    pixel_row_784 : float32 array shape (784,), already divided by 255.
    Returns float32 (784,) — already in the correct space, no change needed.
    """
    return np.asarray(pixel_row_784, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (training only)
# ─────────────────────────────────────────────────────────────────────────────

def augment(img_float32, rng=None):
    """
    Apply randomized augmentation to a 28×28 float32 image (values in [0,1]).
    Used only during training to improve robustness to real-world variation.

    Augmentations applied (all are sign-preserving):
      - Random rotation ±15°
      - Random translation ±2 px
      - Random brightness scale 0.65–1.35
      - Random zoom 0.88–1.10

    Note: Horizontal flip is NOT applied — it would change the handedness
          of the sign and alter meaning for asymmetric letters (e.g. G, Q).
    """
    if rng is None:
        rng = np.random.default_rng()

    img = np.clip(img_float32 * 255, 0, 255).astype(np.uint8)
    h, w = img.shape

    # 1. Rotation
    angle = float(rng.uniform(-15, 15))
    M_r   = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img   = cv2.warpAffine(img, M_r, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    # 2. Translation
    tx = float(rng.uniform(-2.5, 2.5))
    ty = float(rng.uniform(-2.5, 2.5))
    M_t = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M_t, (w, h),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # 3. Brightness
    factor = float(rng.uniform(0.65, 1.35))
    img    = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 4. Zoom (scale then crop/pad back to IMG_SIZExIMG_SIZE)
    scale = float(rng.uniform(0.88, 1.10))
    new_s = max(4, int(round(w * scale)))
    zoomed = cv2.resize(img, (new_s, new_s), interpolation=cv2.INTER_AREA)
    if new_s >= w:
        start_y = (new_s - h) // 2
        start_x = (new_s - w) // 2
        img = zoomed[start_y: start_y + h, start_x: start_x + w]
    else:
        pad_y = (h - new_s) // 2
        pad_x = (w - new_s) // 2
        img = cv2.copyMakeBorder(
            zoomed,
            pad_y, h - new_s - pad_y,
            pad_x, w - new_s - pad_x,
            cv2.BORDER_REPLICATE,
        )

    # Ensure output is exactly IMG_SIZE x IMG_SIZE (guard against off-by-one)
    if img.shape != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32) / 255.0


def augment_batch(img_float32, n_aug=3, rng=None):
    """
    Return the original image + n_aug augmented versions.
    Total returned = n_aug + 1 images.
    """
    if rng is None:
        rng = np.random.default_rng()
    results = [img_float32]
    for _ in range(n_aug):
        results.append(augment(img_float32, rng=rng))
    return results

