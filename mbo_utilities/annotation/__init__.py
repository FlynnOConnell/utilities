"""ROI annotation model + persistence, GUI-free.

The manual ROI widget (``gui/manual_roi.py``) is a thin imgui/pygfx shell
over this package. See README.md for the extraction seam shared with
masknmf-toolbox's ``visualization/imgui`` layer.
"""

from mbo_utilities.annotation.ngff import LabelsZarr
from mbo_utilities.annotation.store import (
    CLASS_COLORS,
    ROI_COLORS,
    UNLABELED,
    RoiLabelStore,
    RoiRecord,
    class_color,
)

__all__ = [
    "CLASS_COLORS",
    "ROI_COLORS",
    "UNLABELED",
    "LabelsZarr",
    "RoiLabelStore",
    "RoiRecord",
    "class_color",
]
