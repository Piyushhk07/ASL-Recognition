"""
mnist_features.py -- Shared feature extraction for MNIST-based ASL model
=========================================================================
Converts a 28x28 grayscale normalized image into a 840-dimensional
feature vector used identically during training AND inference.

Features:
  [A] 784  Raw normalized pixels (flattened 28x28)
  [B]  28  Mean horizontal gradient per row  (texture/edge density)
  [C]  28  Mean vertical gradient per column (texture/edge density)
Total = 840

Usage
-----
from mnist_features import extract_features, NUM_FEATURES
feat = extract_features(flat_784)   # shape (840,) float32
"""

import numpy as np

NUM_FEATURES = 840   # public constant

# MNIST label → ASL letter (J=9 and Z=25 are motion signs, excluded)
LABEL_TO_LETTER = {i: c for i, c in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                   if c not in ('J', 'Z')}

# Reverse map: letter → MNIST label
LETTER_TO_LABEL = {v: k for k, v in LABEL_TO_LETTER.items()}

# Sorted list of present letters (A-Y excluding J)
PRESENT_LETTERS = sorted(LABEL_TO_LETTER.values())


def extract_features(flat_784):
    """
    Extract 840-d features from a flattened 28x28 normalised grayscale image.

    Parameters
    ----------
    flat_784 : array-like, shape (784,)
        Pixel values in [0, 1], row-major order (28×28).

    Returns
    -------
    np.ndarray, shape (840,), dtype float32
    """
    flat = np.asarray(flat_784, dtype=np.float32).reshape(784)
    img  = flat.reshape(28, 28)

    # [A] Raw pixels
    feat_pixels = flat                              # 784

    # [B] Mean absolute horizontal gradient per row
    h_grad = np.abs(np.diff(img, axis=1))          # (28, 27)
    feat_h = h_grad.mean(axis=1).astype(np.float32)  # 28

    # [C] Mean absolute vertical gradient per column
    v_grad = np.abs(np.diff(img, axis=0))          # (27, 28)
    feat_v = v_grad.mean(axis=0).astype(np.float32)  # 28

    out = np.concatenate([feat_pixels, feat_h, feat_v])
    assert len(out) == NUM_FEATURES, f"Expected {NUM_FEATURES}, got {len(out)}"
    return out

