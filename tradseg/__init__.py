"""Standalone traditional (non-neural) segmentation toolkit for the MSD Spleen task.

Self-contained: it consumes the SAME 2-channel preprocessed npy contract
(image[0,1], label) + splits.pkl used by the baseline U-Net, but shares NO code
with the ``my_unet-uncertainty`` folder, so the two pipelines stay independent
while remaining directly comparable (identical data, splits and eval space).
"""

__version__ = "0.1.0"
