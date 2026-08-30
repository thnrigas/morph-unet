"""Method registry (Tier 1 + Tier 2)."""

from .base import Segmenter
from .thresholding import OtsuSegmenter, MultiOtsuSegmenter
from .clustering import KMeansSegmenter, GMMSegmenter
from .region_growing import RegionGrowingSegmenter
from .watershed import WatershedSegmenter
from .levelset import ChanVeseSegmenter, MorphGACSegmenter
from .graphcut import GraphCutSegmenter
from .randomwalker import RandomWalkerSegmenter
from .texture import GaborSegmenter, AmFmSegmenter
from .granulometry import GranulometrySegmenter

REGISTRY = {
    # Tier 1
    "otsu": OtsuSegmenter,
    "multiotsu": MultiOtsuSegmenter,
    "kmeans": KMeansSegmenter,
    "gmm": GMMSegmenter,
    "region_growing": RegionGrowingSegmenter,
    "watershed": WatershedSegmenter,
    # Tier 2
    "chanvese": ChanVeseSegmenter,
    "morphgac": MorphGACSegmenter,
    "graphcut": GraphCutSegmenter,
    "random_walker": RandomWalkerSegmenter,
    # Tier 3 (texture / morphology)
    "gabor": GaborSegmenter,
    "amfm": AmFmSegmenter,
    "granulometry": GranulometrySegmenter,
}

__all__ = [
    "Segmenter", "REGISTRY",
    "OtsuSegmenter", "MultiOtsuSegmenter", "KMeansSegmenter", "GMMSegmenter",
    "RegionGrowingSegmenter", "WatershedSegmenter",
    "ChanVeseSegmenter", "MorphGACSegmenter", "GraphCutSegmenter", "RandomWalkerSegmenter",
    "GaborSegmenter", "AmFmSegmenter", "GranulometrySegmenter",
]
