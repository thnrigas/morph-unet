"""MC-Dropout uncertainty core (faithful copy of my_unet-uncertainty/utilities/mc_dropout.py).

Three moves: (1) keep dropout stochastic at inference; (2) T forward passes -> softmax
probs; (3) reduce to per-pixel uncertainty maps (predictive entropy = total,
mutual information = epistemic, fg-prob variance = epistemic proxy). Plus post-hoc
temperature fitting (Guo et al. 2017). Calibration/plotting helpers are intentionally
omitted here (not needed for the hybrid).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def enable_dropout(model):
    """Put ONLY dropout layers into train() (stay stochastic) while the rest stays eval().
    Returns the number of dropout modules re-enabled (sanity check)."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n += 1
    return n


@torch.no_grad()
def mc_forward(model, data, T, temperature=1.0):
    """T stochastic passes -> softmax probs [T, B, C, H, W]. Assumes enable_dropout() done.
    temperature>1 softens (post-hoc temperature scaling); 1.0 is a no-op."""
    samples = [F.softmax(model(data) / temperature, dim=1) for _ in range(T)]
    return torch.stack(samples, dim=0)


def _entropy(prob, dim):
    return -(prob * torch.log(prob + _EPS)).sum(dim=dim)


def uncertainty_maps(probs):
    """probs [T, B, C, H, W] -> dict of maps (mean_prob, pred, entropy, expected_entropy,
    mutual_info, fg_var)."""
    mean_prob = probs.mean(dim=0)                                 # [B, C, H, W]
    pred = mean_prob.argmax(dim=1)                                # [B, H, W]
    predictive_entropy = _entropy(mean_prob, dim=1)              # total
    expected_entropy = _entropy(probs, dim=2).mean(dim=0)        # aleatoric
    mutual_info = (predictive_entropy - expected_entropy).clamp_min(0.0)   # epistemic
    fg_prob = 1.0 - probs[:, :, 0]                                # P(not background) [T,B,H,W]
    fg_var = fg_prob.var(dim=0, unbiased=False)                  # [B, H, W]
    return {
        "mean_prob": mean_prob, "pred": pred, "entropy": predictive_entropy,
        "expected_entropy": expected_entropy, "mutual_info": mutual_info, "fg_var": fg_var,
    }


def fit_temperature(logits, targets, max_iter=200, lr=0.1):
    """Fit scalar T>0 minimising NLL of softmax(logits/T) (pass FOREGROUND pixels only).
    Does not change argmax -> Dice unaffected. logits [N,C], targets [N] long -> float T."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    ce = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = ce(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())
