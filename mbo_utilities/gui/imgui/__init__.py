"""Shared imgui widgets for the ROI viewers.

Ported from ``masknmf.visualization.imgui`` so the viewer keeps working
against any masknmf branch: the label set, the sortable ROI table, stroke
capture, the label/progress panels and the summary image popup all live
here now. Styling comes from ``mbo_utilities.gui._theme``.
"""

from mbo_utilities.gui.imgui.crop import context_crop, crop_origin, crop_slices
from mbo_utilities.gui.imgui.draw import StrokeDrawer
from mbo_utilities.gui.imgui.labels import (
    LABEL_COLORS,
    LABEL_KEYS,
    UNLABEL_ALL,
    UNLABELED,
    LabelSet,
)
from mbo_utilities.gui.imgui.movie_player import MoviePlayer
from mbo_utilities.gui.imgui.panels import (
    draw_keybinds_popup,
    draw_label_buttons,
    draw_label_editor,
    draw_progress,
)
from mbo_utilities.gui.imgui.summary import SummaryImageViewer
from mbo_utilities.gui.imgui.table import (
    FILTER_ALL,
    RoiOrder,
    RowAction,
    draw_filter_row,
    draw_label_filter,
    draw_range_filter,
    draw_roi_table,
)

__all__ = [
    "FILTER_ALL",
    "LABEL_COLORS",
    "LABEL_KEYS",
    "LabelSet",
    "MoviePlayer",
    "RoiOrder",
    "RowAction",
    "StrokeDrawer",
    "SummaryImageViewer",
    "UNLABELED",
    "UNLABEL_ALL",
    "context_crop",
    "crop_origin",
    "crop_slices",
    "draw_filter_row",
    "draw_keybinds_popup",
    "draw_label_buttons",
    "draw_label_editor",
    "draw_label_filter",
    "draw_progress",
    "draw_range_filter",
    "draw_roi_table",
]
