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
  - Histogram equalization (cv2.equalizeHist) applied to BOTH training AND
    inference. This normalises contrast so that variable webcam lighting and
    different backgrounds produce pixel distributions closer to training data.
  - NO CLAHE: global equaliseHist is deterministic and symmetric — it is
    applied identically at training (MNIST rows) and inference (webcam crop).
    CLAHE is tile-local and was only applied at inference previously, which
    caused a distribution mismatch.
  - Consistent resize: INTER_AREA downscale for both train and inference.
  - Normalization: divide by 255 only (after equalisation). No Z-score.
  - Augmentation: random rotation / brightness / shift only during TRAINING.
"""

import cv2
import numpy as np

IMG_SIZE = 28   # Sign MNIST image dimensions


# ─────────────────────────────────────────────────────────────────────────────
# Core preprocessing (train + inference — must be identical)
# ─────────────────────────────────────────────────────────────────────────────

def _equalize_and_normalize(gray_uint8):
    """
    Apply global histogram equalization then normalise to [0, 1].
    Input : uint8 grayscale image (any size).
    Output: float32 array, same shape, values in [0, 1].

    Histogram equalization spreads the pixel intensity histogram to the full
    [0, 255] range.  This makes the model invariant to:
      - Absolute brightness (dim room vs bright room)
      - Low-contrast scenes (hand against wall)
    Applying it identically in training and inference ensures no mismatch.
    """
    equalized = cv2.equalizeHist(gray_uint8)
    return equalized.astype(np.float32) / 255.0


def to_28x28(gray_crop):
    """
    Resize a grayscale crop to 28x28 using INTER_AREA,
    apply histogram equalization, and normalise to [0, 1].
    Expects a uint8 or float32 numpy array.
    Returns float32 (28, 28) with values in [0, 1].
    """
    if gray_crop.dtype != np.uint8:
        gray_crop = np.clip(gray_crop * 255, 0, 255).astype(np.uint8)
    resized = cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    return _equalize_and_normalize(resized)


def preprocess_bgr_crop(bgr_crop):
    """
    Full preprocessing for a BGR webcam hand crop:
      1. Convert to grayscale
      2. Resize to 28x28 (INTER_AREA)
      3. Histogram equalisation (same as applied to training images)
      4. Normalise to [0, 1]

    This now matches exactly what train_model.py applies to MNIST pixels.
    """
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return to_28x28(gray)


def preprocess_mnist_row(pixel_row_784):
    """
    Preprocess one row from the Sign MNIST CSV for training.
    pixel_row_784 : float32 array shape (784,), values in [0, 1] (raw /255).

    Applies the same histogram equalization that inference uses, so the
    model is trained on equalized images and sees equalized images at
    inference -- perfect consistency.

    Returns float32 (784,) equalized and normalized.
    """
    arr = np.asarray(pixel_row_784, dtype=np.float32)
    # Convert back to uint8 28x28 to apply equalizeHist
    img_u8 = np.clip(arr * 255, 0, 255).astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE)
    equalized = _equalize_and_normalize(img_u8)
    return equalized.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (training only)
# ─────────────────────────────────────────────────────────────────────────────

def augment(img_float32, rng=None):
    """
    Apply randomized augmentation to a 28x28 float32 image (values in [0,1]).
    Used only during training to improve robustness to real-world variation.

    NOTE: augmentation is applied AFTER histogram equalization (the input
    image is already equalized).  The brightness augmentation simulates
    residual contrast variation that equalization does not fully remove.

    Augmentations applied (all are sign-preserving):
      - Random rotation +-15 degrees
      - Random translation +-2 px
      - Random brightness scale 0.75-1.25 (narrowed: equalization reduces need)
      - Random zoom 0.88-1.10

    Note: Horizontal flip is NOT applied -- it would change the handedness
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

    # 3. Brightness (narrowed range since equalization handles gross contrast)
    factor = float(rng.uniform(0.75, 1.25))
    img    = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 4. Zoom (scale then crop/pad back to IMG_SIZExIMG_SIZE)
    scale  = float(rng.uniform(0.88, 1.10))
    new_s  = max(4, int(round(w * scale)))
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
