"""
features.py  --  ASL Vision shared feature engineering
=======================================================
Converts raw 63-d MediaPipe landmarks into a rich, scale- and
translation-invariant feature vector (~99 dims) that is far more
discriminative for ASL letter classification than raw coords.

Used identically by train_model.py and app.py so that training
and inference always operate on the same representation.
"""

import numpy as np

# Landmark indices (MediaPipe Hands)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
IDX_MCP,  IDX_PIP,  IDX_DIP,  IDX_TIP  = 5, 6, 7, 8
MID_MCP,  MID_PIP,  MID_DIP,  MID_TIP  = 9, 10, 11, 12
RNG_MCP,  RNG_PIP,  RNG_DIP,  RNG_TIP  = 13, 14, 15, 16
PNK_MCP,  PNK_PIP,  PNK_DIP,  PNK_TIP  = 17, 18, 19, 20

FINGERTIP_IDS   = [THUMB_TIP, IDX_TIP, MID_TIP, RNG_TIP, PNK_TIP]
MCP_IDS         = [THUMB_MCP, IDX_MCP, MID_MCP, RNG_MCP, PNK_MCP]

# Each finger: list of joint landmark ids (base -> tip)
FINGER_CHAINS = [
    [THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP],       # thumb
    [IDX_MCP,  IDX_PIP,  IDX_DIP,  IDX_TIP],           # index
    [MID_MCP,  MID_PIP,  MID_DIP,  MID_TIP],           # middle
    [RNG_MCP,  RNG_PIP,  RNG_DIP,  RNG_TIP],           # ring
    [PNK_MCP,  PNK_PIP,  PNK_DIP,  PNK_TIP],           # pinky
]

# MediaPipe hand connections for skeleton drawing (used by JS too)
HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
    [5,9],[9,13],[13,17],
]


def _safe_norm(v, eps=1e-6):
    n = np.linalg.norm(v)
    return v / (n + eps), n


def _angle_cos(a, b, c):
    """Cosine of angle at vertex b formed by a-b-c."""
    v1, n1 = _safe_norm(a - b)
    v2, n2 = _safe_norm(c - b)
    return float(np.clip(np.dot(v1, v2), -1.0, 1.0))


def extract_features(raw_63):
    """
    Extract a 99-dimensional feature vector from 63 raw landmark coords.

    Features are:
      [A] 63  -- normalized (x,y,z) relative to wrist, scaled by hand size
      [B]  5  -- fingertip distances from wrist (normalized)
      [C]  5  -- finger extension ratios (tip / MCP distance from wrist)
      [D] 10  -- inter-fingertip pairwise distances
      [E] 10  -- joint bend angles (cos) along each finger
      [F]  3  -- thumb-tip to index-tip displacement (3-d)
      [G]  3  -- palm normal direction (cross-product of palm vectors)
    Total = 99

    Parameters
    ----------
    raw_63 : array-like, shape (63,)
        Flattened [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20].

    Returns
    -------
    np.ndarray, shape (99,), dtype float32
    """
    pts = np.array(raw_63, dtype=np.float64).reshape(21, 3)

    # ── A: Translate to wrist origin, scale by hand size ─────────────────────
    wrist      = pts[WRIST].copy()
    pts_n      = pts - wrist                                   # translate
    hand_size  = np.linalg.norm(pts_n[MID_MCP]) + 1e-6        # distance wrist→mid-MCP
    pts_n     /= hand_size                                     # scale

    feat_coords = pts_n.flatten()                              # 63 features

    # ── B: Fingertip distances from wrist ─────────────────────────────────────
    feat_tip_dist = np.array(
        [np.linalg.norm(pts_n[t]) for t in FINGERTIP_IDS]
    )                                                          # 5 features

    # ── C: Extension ratio = tip_dist / MCP_dist ──────────────────────────────
    feat_extension = np.array([
        np.linalg.norm(pts_n[tip]) / (np.linalg.norm(pts_n[mcp]) + 1e-6)
        for tip, mcp in zip(FINGERTIP_IDS, MCP_IDS)
    ])                                                         # 5 features

    # ── D: Inter-fingertip pairwise distances (10 pairs) ─────────────────────
    tip_pts = pts_n[FINGERTIP_IDS]
    feat_tip_pairs = np.array([
        np.linalg.norm(tip_pts[i] - tip_pts[j])
        for i in range(5) for j in range(i+1, 5)
    ])                                                         # 10 features

    # ── E: Joint bend angles along each finger (2 bends × 5 fingers = 10) ────
    feat_angles = []
    for chain in FINGER_CHAINS:
        for k in range(len(chain) - 2):
            feat_angles.append(_angle_cos(pts_n[chain[k]],
                                          pts_n[chain[k+1]],
                                          pts_n[chain[k+2]]))
    feat_angles = np.array(feat_angles)                       # 10 features

    # ── F: Thumb-tip to index-tip 3-d displacement ───────────────────────────
    feat_thumb_idx = pts_n[THUMB_TIP] - pts_n[IDX_TIP]       # 3 features

    # ── G: Palm normal (cross product of two palm vectors) ───────────────────
    v_palm_h = pts_n[IDX_MCP] - pts_n[PNK_MCP]               # horizontal
    v_palm_v = pts_n[MID_MCP] - pts_n[WRIST]                 # vertical
    palm_normal, _ = _safe_norm(np.cross(v_palm_h, v_palm_v))
    feat_palm_norm = palm_normal                              # 3 features

    # ── Concatenate all ────────────────────────────────────────────────────────
    out = np.concatenate([
        feat_coords,       # 63
        feat_tip_dist,     #  5
        feat_extension,    #  5
        feat_tip_pairs,    # 10
        feat_angles,       # 10
        feat_thumb_idx,    #  3
        feat_palm_norm,    #  3
    ]).astype(np.float32)

    assert len(out) == 99, f"Expected 99 features, got {len(out)}"
    return out


NUM_FEATURES = 99   # public constant for other modules

