"""UNetPredictor — the communication layer between the trained U-Net and the traditional
hybrid. Loads a checkpoint and predicts on a preprocessed image volume (Z,H,W) in [0,1]
(= channel 0 of the (2,Z,S,S) npy). Direct per-slice forward (no batchgenerators). Returns
segmentation, softmax probability, and (MC-Dropout) uncertainty maps, all per-volume.

Config-free on purpose: pass the checkpoint path + params explicitly so the hybrid code
(in the tradseg root) can `import predictor` after putting unets/ on sys.path.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

from arch import MCDropoutUNet
import mc_dropout as MC


def pick_device(device=None):
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class UNetPredictor:
    def __init__(self, ckpt, num_classes=2, in_channels=1, dropout_p=0.4, device=None):
        self.device = pick_device(device)
        self.num_classes = int(num_classes)
        self.model = MCDropoutUNet(num_classes=num_classes, in_channels=in_channels,
                                   dropout_p=dropout_p).to(self.device)
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

    def _slice_batches(self, image_vol, batch):
        z = image_vol.shape[0]
        for z0 in range(0, z, batch):
            z1 = min(z0 + batch, z)
            data = torch.from_numpy(np.ascontiguousarray(image_vol[z0:z1, None])).float().to(self.device)
            yield z0, z1, data

    @torch.no_grad()
    def predict(self, image_vol, batch=8):
        """Deterministic (dropout OFF) -> {'seg' (Z,H,W) int16, 'prob' (Z,C,H,W) float32}."""
        self.model.eval()
        z, h, w = image_vol.shape
        prob = np.empty((z, self.num_classes, h, w), np.float32)
        for z0, z1, data in self._slice_batches(image_vol, batch):
            prob[z0:z1] = torch.softmax(self.model(data), dim=1).cpu().numpy()
        return {"seg": prob.argmax(1).astype(np.int16), "prob": prob}

    @torch.no_grad()
    def predict_mc(self, image_vol, T=30, temperature=1.0, batch=8):
        """MC-Dropout (T stochastic passes) -> per-volume numpy maps:
        {'seg','mean_prob','entropy','mutual_info','fg_var'}. `seg` is argmax of the mean prob."""
        self.model.eval()
        if MC.enable_dropout(self.model) == 0:
            raise RuntimeError("no dropout layers found -> MC dropout would be deterministic")
        z, h, w = image_vol.shape
        out = {
            "seg": np.empty((z, h, w), np.int16),
            "mean_prob": np.empty((z, self.num_classes, h, w), np.float32),
            "entropy": np.empty((z, h, w), np.float32),
            "mutual_info": np.empty((z, h, w), np.float32),
            "fg_var": np.empty((z, h, w), np.float32),
        }
        for z0, z1, data in self._slice_batches(image_vol, batch):
            probs = MC.mc_forward(self.model, data, T, temperature=temperature)   # [T,b,C,H,W]
            m = MC.uncertainty_maps(probs)
            out["seg"][z0:z1] = m["pred"].cpu().numpy().astype(np.int16)
            out["mean_prob"][z0:z1] = m["mean_prob"].cpu().numpy()
            out["entropy"][z0:z1] = m["entropy"].cpu().numpy()
            out["mutual_info"][z0:z1] = m["mutual_info"].cpu().numpy()
            out["fg_var"][z0:z1] = m["fg_var"].cpu().numpy()
        return out
