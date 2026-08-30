"""3D post-processing shared by every method.

The single most important step for the Spleen: a per-slice 2D method produces many
false positives on the ~60-90% of axial slices with no spleen, so we clean the
STACKED 3D mask (largest connected component / small-object removal / hole filling).
This is the traditional-side analogue of the U-Net's train-time foreground slice
curation, applied at inference where both pipelines are scored on every slice.
"""

import numpy as np
from scipy import ndimage as ndi


def _structure(ndim, connectivity):
    return ndi.generate_binary_structure(ndim, connectivity)


def keep_largest_cc(mask, connectivity=1):
    """Keep only the largest connected component (6-neighbour by default in 3D)."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=_structure(mask.ndim, connectivity))
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def keep_k_largest_cc(mask, k=1, connectivity=1):
    mask = np.asarray(mask, bool)
    if not mask.any() or k < 1:
        return mask
    lab, n = ndi.label(mask, structure=_structure(mask.ndim, connectivity))
    if n <= k:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep_ids = set(np.argsort(sizes)[::-1][:k].tolist())
    return np.isin(lab, list(keep_ids))


def remove_small_objects(mask, min_size, connectivity=1):
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=_structure(mask.ndim, connectivity))
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(sizes.shape[0], bool)
    keep[1:] = sizes[1:] >= min_size
    return keep[lab]


def fill_holes(mask):
    """Fill fully-enclosed 3D cavities. NOTE: on a Z==1 volume every voxel touches
    the degenerate z-border, so nothing fills -- use fill_holes_2d for stacked slices."""
    return ndi.binary_fill_holes(np.asarray(mask, bool))


def fill_holes_2d(mask):
    """Fill holes independently within each axial slice (right for stacked 2D masks)."""
    mask = np.asarray(mask, bool)
    out = np.empty_like(mask)
    for z in range(mask.shape[0]):
        out[z] = ndi.binary_fill_holes(mask[z])
    return out


def binary_closing(mask, iterations=1, connectivity=1):
    mask = np.asarray(mask, bool)
    return ndi.binary_closing(mask, structure=_structure(mask.ndim, connectivity),
                              iterations=iterations)


def binary_opening(mask, iterations=1, connectivity=1):
    mask = np.asarray(mask, bool)
    return ndi.binary_opening(mask, structure=_structure(mask.ndim, connectivity),
                              iterations=iterations)


def select_component_by_prior(mask, prior, connectivity=1, reduce="mean"):
    """Keep the connected component that best matches a spatial prior map.

    prior : float array, same shape as mask; higher = more likely foreground.
    reduce: 'mean' (average prior inside the component) or 'sum' (mass-weighted).
    Used by the AUTO regime to pick the spleen among several soft-tissue blobs.
    """
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    lab, n = ndi.label(mask, structure=_structure(mask.ndim, connectivity))
    if n <= 1:
        return mask
    best_id, best_score = 0, -np.inf
    for i in range(1, n + 1):
        comp = lab == i
        score = prior[comp].mean() if reduce == "mean" else prior[comp].sum()
        if score > best_score:
            best_score, best_id = score, i
    return lab == best_id


def select_component_by_overlap(mask, reference, connectivity=1):
    """Keep the connected component with the largest overlap with a binary reference.
    Used by the ORACLE regime to pick the component that best matches the GT."""
    return select_component_by_prior(mask, np.asarray(reference, dtype=float),
                                     connectivity=connectivity, reduce="sum")


def clean(mask, min_size=50, largest_cc=True, fill=True, fill_per_slice=True,
          closing_iters=0, connectivity=1):
    """Convenience default pipeline: remove specks -> (optional) largest CC -> fill holes.
    fill_per_slice=True fills holes slice-by-slice (2D), the right choice for masks
    built by stacking per-slice 2D predictions."""
    out = remove_small_objects(mask, min_size, connectivity) if min_size else np.asarray(mask, bool)
    if closing_iters:
        out = binary_closing(out, closing_iters, connectivity)
    if largest_cc:
        out = keep_largest_cc(out, connectivity)
    if fill:
        out = fill_holes_2d(out) if fill_per_slice else fill_holes(out)
    return out
