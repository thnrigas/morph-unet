"""Common segmenter interface for every traditional method.

Contract:
  input  image : (Z, H, W) float32 in [0, 1]  -- channel 0 of the preprocessed npy;
                 air / out-of-body padding == 0 by construction.
  output mask  : (Z, H, W) bool                -- True = spleen candidate (RAW, before
                 3D post-processing / component selection, which the runner applies).

`seeds` (optional): dict with 'fg'/'bg' boolean volumes for seeded methods
(region growing, watershed, graph cuts). Intensity-only methods ignore it.

Segmenters must NEVER look at the ground truth for their prediction; the runner is
responsible for deriving oracle seeds from the GT when the oracle regime is used.
"""

from abc import ABC, abstractmethod

import numpy as np


class Segmenter(ABC):
    name = "base"
    tier = 1                    # 1 = baseline, 2 = level sets / graph cut / random walker, ...
    requires_seeds = False
    seed_type = None            # 'points' (region growing) | 'markers' (watershed / graph cut / ...)

    @abstractmethod
    def segment_volume(self, image, seeds=None):
        """Return a (Z, H, W) boolean mask for the given image volume."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


def apply_per_slice(fn, image):
    """Run a 2D segmentation callable fn(img2d)->bool mask over each axial slice."""
    out = np.zeros(image.shape, dtype=bool)
    for z in range(image.shape[0]):
        out[z] = fn(image[z])
    return out


def body_mask(image, thr=0.0):
    """Voxels inside the body (air / padding == 0 after CT windowing)."""
    return np.asarray(image) > thr
