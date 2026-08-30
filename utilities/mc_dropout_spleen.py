#
# Backward-compat shim.
#
# The corrected calibration (per-class over pred==c, temperature safety rail) now
# lives in utilities/mc_calibration.py so BOTH datasets share one source of truth.
# This module is kept only so the existing spleen scripts keep importing
# `SegCalibrationSpleen` / `fit_temperature_safe` unchanged (byte-identical behaviour).
#

from utilities.mc_calibration import (  # noqa: F401
    SegCalibration as SegCalibrationSpleen,
    fit_temperature_safe,
    save_calibration_figure,
    fit_temperature,
)
