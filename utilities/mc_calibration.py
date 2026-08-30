#
# Shared, CORRECTED calibration for the MC-Dropout uncertainty track (both datasets)
#
# Single source of truth for the two policy fixes we validated on the Spleen and now
# apply to the Hippocampus too, for consistency:
#
#   1. per-class reliability over PREDICTED-positive pixels (pred==c), NOT the
#      foreground union ROI (gt>0 | pred>0). The union ROI drops every missed-organ
#      pixel (gt==c, p_c<0.5) into the low-p_c bins with event==1 -> a spurious
#      "accuracy ~ 1 at low confidence" SELECTION artifact. pred==c is an unbiased
#      (precision-style) reliability, and for a binary task it ~matches foreground.
#
#   2. temperature fitting with a SAFETY RAIL: accept T only in [0.5, 10]; a
#      pathological optimum outside that range (a confidently-wrong model drives
#      T -> inf, softmax -> uniform 0.5, zero resolution) falls back to T=1.0 and
#      warns, keeping the raw but informative uncertainty. (The fit itself is done on
#      gt>0 by the callers, so it does not depend on the model's own predictions.)
#
# The heavy machinery (_Bins, curves(), summary(), the figures, fit_temperature,
# mc_forward, ...) lives in utilities/mc_dropout.py and is reused by import.
#

import torch

from utilities.mc_dropout import (  # noqa: F401  (several are re-exported for callers)
    SegCalibration as _BaseSegCalibration,
    save_calibration_figure,
    save_uncertainty_png,
    fit_temperature,
    enable_dropout,
    mc_forward,
    uncertainty_maps,
)


# ---- temperature-scaling safety rail (Guo et al. 2017 + a sane-range guard) --
_T_LO, _T_HI = 0.5, 10.0


def fit_temperature_safe(logits, targets, t_lo=_T_LO, t_hi=_T_HI):
    """Fit a scalar temperature, then REJECT a pathological optimum.

    A single temperature can only globally sharpen (T<1) or flatten (T>1). On a
    confidently-wrong / degenerate model the NLL is minimised by T -> inf, i.e.
    softmax -> uniform (every probability 0.5, entropy = ln2, ZERO resolution),
    which destroys the uncertainty output. So if the fitted T lands outside
    [t_lo, t_hi] we fall back to T=1.0 (a no-op) and warn, keeping the raw but
    INFORMATIVE uncertainty instead of collapsing it. Returns (temperature, accepted).
    """
    t = fit_temperature(logits, targets)
    if not (t_lo <= t <= t_hi):
        print(f"[calibrate] fitted T={t:.4g} outside [{t_lo}, {t_hi}] -> model is "
              f"confidently wrong / degenerate on the fit set; falling back to T=1.0 "
              f"(raw uncertainty kept, post-hoc calibration skipped).")
        return 1.0, False
    return t, True


class SegCalibration(_BaseSegCalibration):
    """Corrected SegCalibration: per-class reliability over predicted-positive pixels.

    Identical to the base (global + foreground curves, _Bins, curves(), summary(),
    figure) EXCEPT the per-class bins use pred==c instead of the foreground union
    ROI -> no false-negative artifact."""

    @torch.no_grad()
    def update(self, mean_prob, gt):
        conf_top, pred = mean_prob.max(dim=1)                     # [B, H, W]
        self.glob.add(conf_top, (pred == gt).float())            # reference: all pixels

        roi = (gt > 0) | (pred > 0)                              # foreground union (top-label)
        if roi.any():
            self.fg.add(conf_top[roi], (pred[roi] == gt[roi]).float())

        # per-class: PREDICTED-c pixels only (pred==c <=> p_c is the max) -> p_c vs
        # (gt==c). Excludes the missed-c (gt==c, p_c<0.5) pixels that caused the
        # low-confidence artifact; no gt-conditioning of the low-p_c bins.
        for c, bins in self.cls.items():
            m = (pred == c)
            if m.any():
                bins.add(mean_prob[:, c][m], (gt[m] == c).float())
