"""Shared texture / morphology feature extractors for the Tier-3 methods (Ch.13 Maragos).

All operate on a single 2D slice (float image in [0,1]) and are used by the feature-
clustering segmenters in methods/texture.py and methods/granulometry.py. Heavy libs
(scikit-image) are imported lazily so importing this module stays cheap.
"""

import numpy as np


# --------------------------- anisotropic diffusion ---------------------------
def perona_malik(img2d, iterations=5, kappa=0.05, gamma=0.15):
    """Perona-Malik anisotropic diffusion: edge-preserving smoothing (Ch.17 nonlinear
    scale-space). img in [0,1]; kappa = edge-contrast threshold; gamma < 0.25 for stability."""
    u = img2d.astype(np.float32).copy()
    for _ in range(int(iterations)):
        dn = np.roll(u, -1, 0) - u
        ds = np.roll(u, 1, 0) - u
        de = np.roll(u, -1, 1) - u
        dw = np.roll(u, 1, 1) - u
        cn = np.exp(-(dn / kappa) ** 2)
        cs = np.exp(-(ds / kappa) ** 2)
        ce = np.exp(-(de / kappa) ** 2)
        cw = np.exp(-(dw / kappa) ** 2)
        u = u + gamma * (cn * dn + cs * ds + ce * de + cw * dw)
    return u


# ------------------------------- Gabor bank ----------------------------------
def gabor_bank(frequencies=(0.15, 0.3), n_orient=4):
    """List of (frequency, theta) pairs for a radial Gabor filterbank."""
    thetas = [np.pi * i / n_orient for i in range(n_orient)]
    return [(float(f), float(t)) for f in frequencies for t in thetas]


def gabor_features(img2d, bank):
    """Gabor energy (amplitude envelope) per (frequency, orientation) -> (B, H, W)."""
    from skimage.filters import gabor
    feats = []
    for f, t in bank:
        re, im = gabor(img2d, frequency=f, theta=t)
        feats.append(np.sqrt(re.astype(np.float32) ** 2 + im.astype(np.float32) ** 2))
    return np.stack(feats)


# ------------------------ Teager–Kaiser energy (Ch.13) -----------------------
def teager_energy_2d(img2d):
    """Discrete 2D multidimensional energy operator (Maragos-Bovik-Quatieri):
    Phi_d = 2 f^2 - f(m-1,n)f(m+1,n) - f(m,n-1)f(m,n+1)."""
    f = img2d.astype(np.float32)
    fx1 = np.roll(f, 1, 0); fx2 = np.roll(f, -1, 0)
    fy1 = np.roll(f, 1, 1); fy2 = np.roll(f, -1, 1)
    return 2.0 * f * f - fx1 * fx2 - fy1 * fy2


# --------------------- granulometry / local pattern spectrum -----------------
def granulometry_features(img2d, radii=(1, 2, 4, 8)):
    """Per-pixel multiscale morphological residuals = local pattern spectrum (Ch.13):
    white top-hat (bright peaks <= r) and black top-hat (dark valleys <= r) per radius."""
    from skimage.morphology import disk, opening, closing
    feats = []
    for r in radii:
        se = disk(int(r))
        feats.append(img2d - opening(img2d, se))    # white top-hat
        feats.append(closing(img2d, se) - img2d)    # black top-hat
    return np.stack(feats).astype(np.float32)
