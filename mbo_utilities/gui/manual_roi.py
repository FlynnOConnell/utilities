"""Manual ROI drawing, labelling and export.

Opened with ``mbo <path> --widget manualroi``. Attaches to PreviewDataWidget
as an extra "ROI" tab so the Preview / Signal Quality / Run tabs and all of
the windowing controls stay available.

Arm "Add ROI", drag a closed stroke around a cell, release and the enclosed
pixels become a mask. ROIs live in one uint16 label image so they can never
overlap. Each ROI can be given a class label; 1-9 assign, 0 clears.

The drawing, mask, overlay, label and table machinery is shared with
masknmf's classification GUI via ``masknmf.visualization.imgui``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from imgui_bundle import imgui, imgui_ctx
from masknmf.visualization.imgui import (
    LabelImage,
    LabelSet,
    LabelStore,
    RoiOrder,
    StrokeDrawer,
    draw_filter_row,
    draw_keybinds_popup,
    draw_label_buttons,
    draw_label_editor,
    draw_progress,
    draw_roi_table,
    label_image_rgba,
)
from masknmf.visualization.imgui.overlay import SELECTED_ALPHA

from mbo_utilities import log
from mbo_utilities.gui._imgui_helpers import selected_button_style, set_tooltip

__all__ = ["ManualRoiWidget", "SELECTED_OPACITY"]

SELECTED_OPACITY = SELECTED_ALPHA

COLUMNS = ("id", "label", "area")

KEYBINDS = (
    ("a", "arm / disarm drawing"),
    ("esc", "stop drawing"),
    ("ctrl+z", "undo last ROI"),
    ("up / down", "previous / next ROI"),
    ("1-9", "assign label"),
    ("0", "clear label"),
    ("m", "toggle masks"),
    ("o", "toggle outlines"),
)

# one saturated color per ROI; hues shuffled so neighbours contrast
_HSV = np.zeros((180, 1, 3), np.uint8)
_HSV[:, 0, 0] = np.random.default_rng(0).permutation(180)
_HSV[:, 0, 1:] = 255
_IDENTITY_COLORS = cv2.cvtColor(_HSV, cv2.COLOR_HSV2RGB).reshape(-1, 3) / 255.0


class ManualRoiWidget:
    """Freehand ROI painting and labelling over a ``MboNDViewer``."""

    def __init__(self, iw, fpath=None):
        self.iw = iw
        self.fpath = Path(fpath) if fpath is not None else None
        self.logger = log.get("gui.manual_roi")

        self.subplot = iw.figure[0, 0]
        ny, nx = iw.graphics[0].data.value.shape[:2]

        self.masks = LabelImage((ny, nx))
        self.classes = LabelSet(0, ("cell", "not cell"))
        self.store = LabelStore(npz_path=str(self._output_path("manual_labels.npz")))
        self.order = RoiOrder({"area": np.zeros(0, np.int64)}, self.classes.labels, 0)
        self.order.set_range_column("area")

        self.selected = -1
        self.status = "press Add ROI to start"
        self.new_label = ""
        self.keybinds_open = False
        self.scroll_to_selection = False

        self.show_masks = True
        self.show_outlines = True
        self.opacity = 0.45

        self.overlay = self.subplot.add_image(
            np.zeros((ny, nx, 4), np.uint8),
            name="manual_roi_overlay",
            alpha_mode="blend",
            offset=(0, 0, 1),
        )
        # keep the overlay out of picking so the tooltip still reports the
        # image intensity under the cursor
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False

        self.drawer = StrokeDrawer(self.subplot, self.add_roi, self.pick_roi)

    @property
    def labels(self) -> np.ndarray:
        """uint16 label image; 0 is background, ROI i is i + 1."""
        return self.masks.labels

    @property
    def counts(self) -> list:
        return self.masks.counts

    @property
    def ny(self) -> int:
        return self.masks.ny

    @property
    def nx(self) -> int:
        return self.masks.nx

    @property
    def drawing(self) -> bool:
        return self.drawer.armed

    @property
    def stroke(self) -> list:
        return self.drawer.stroke

    @property
    def stroke_line(self):
        return self.drawer.line

    def _output_path(self, name: str) -> Path:
        base = Path.cwd() if self.fpath is None else self.fpath
        return (base.parent if base.suffix else base) / name

    @property
    def n_rois(self) -> int:
        return len(self.masks)

    def _resync(self):
        self.classes.resize(self.n_rois)
        self.order.columns = {"area": self.masks.areas()}
        self.order.labels = self.classes.labels
        self.order.n_items = self.n_rois
        self.order.set_range_column("area")
        self.order.rebuild()

    def add_roi(self, stroke):
        index = self.masks.add(stroke)
        if index < 0:
            self.status = self.masks.last_error or "stroke rejected"
            return
        self._resync()
        self.select_roi(index)

    def pick_roi(self, row: int, col: int):
        self.select_roi(self.masks.at(row, col))

    def select_roi(self, index: int):
        self.selected = index if 0 <= index < self.n_rois else -1
        self.scroll_to_selection = True
        if self.selected < 0:
            self.status = f"{self.n_rois} ROIs"
        else:
            self.order.goto(self.selected)
            self.status = (
                f"ROI {self.selected + 1}: {self.masks.counts[self.selected]} px"
            )
        self.refresh_overlay()

    def delete_roi(self, index: int):
        if not self.masks.delete(index):
            return
        self._resync()
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index + 1}"

    def clear(self):
        self.masks.clear()
        self._resync()
        self.select_roi(-1)
        self.status = "cleared"

    def label_selected(self, label_index: int):
        if self.selected < 0:
            return
        self.classes.assign([self.selected], label_index)
        self.save_labels()
        self.order.rebuild()
        self.refresh_overlay()

    def _colors(self) -> np.ndarray:
        """Class colour where labelled, identity colour otherwise."""
        colors = np.zeros((max(self.n_rois, 1), 3), np.float32)
        for i in range(self.n_rois):
            label = int(self.classes.labels[i])
            colors[i] = (
                self.classes.color(label) if label >= 0
                else _IDENTITY_COLORS[i % len(_IDENTITY_COLORS)]
            )
        return colors

    def refresh_overlay(self):
        self.overlay.visible = self.show_masks or self.show_outlines
        if not self.overlay.visible:
            return
        self.overlay.data = label_image_rgba(
            self.masks.labels,
            colors=self._colors(),
            alpha=self.opacity,
            selected=self.selected,
            show_masks=self.show_masks,
            show_outlines=self.show_outlines,
            edges=self.masks.edges(),
        )

    def save_labels(self):
        if self.n_rois:
            self.store.save(self.classes.names, self.classes.labels)

    def save(self):
        out = self._output_path("manual_masks.npy")
        np.save(out, self.masks.labels)
        self.save_labels()
        self.status = f"saved to {out.name}"
        self.logger.info(f"saved {self.n_rois} ROIs to {out}")

    def handle_keys(self):
        io = imgui.get_io()
        if io.want_text_input:
            return
        if imgui.is_key_pressed(imgui.Key.a, False):
            self.set_drawing(not self.drawer.armed)
        if self.drawer.armed and imgui.is_key_pressed(imgui.Key.escape):
            self.set_drawing(False)
        if io.key_ctrl and imgui.is_key_pressed(imgui.Key.z, False):
            self.delete_roi(self.n_rois - 1)
        if imgui.is_key_pressed(imgui.Key.m, False):
            self.show_masks = not self.show_masks
            self.refresh_overlay()
        if imgui.is_key_pressed(imgui.Key.o, False):
            self.show_outlines = not self.show_outlines
            self.refresh_overlay()
        if imgui.is_key_pressed(imgui.Key.up_arrow) and self.order.step(-1):
            self.select_roi(self.order.current)
        if imgui.is_key_pressed(imgui.Key.down_arrow) and self.order.step(1):
            self.select_roi(self.order.current)
        hotkey = self.classes.hotkey_pressed()
        if hotkey is not None:
            self.label_selected(hotkey)

    def set_drawing(self, on: bool):
        self.drawer.arm(on)
        self.status = (
            "drag a closed stroke around a cell" if on else f"{self.n_rois} ROIs"
        )

    def draw_tab(self):
        """The ROI tab body: tools, labels, then the ROI table."""
        self.handle_keys()

        with selected_button_style(self.drawer.armed):
            if imgui.button("Add ROI"):
                self.set_drawing(not self.drawer.armed)
        set_tooltip(
            "Drag a closed stroke around a cell, release to fill it. "
            "'a' toggles, esc stops, ctrl+Z undoes.",
            show_mark=False,
        )
        imgui.same_line()
        if imgui.button("Undo"):
            self.delete_roi(self.n_rois - 1)
        imgui.same_line()
        if imgui.button("Clear"):
            self.clear()
        imgui.same_line()
        if imgui.button("Save"):
            self.save()
        imgui.same_line()
        if imgui.button("keybinds"):
            self.keybinds_open = True

        masks_changed, self.show_masks = imgui.checkbox("Masks", self.show_masks)
        imgui.same_line()
        outlines_changed, self.show_outlines = imgui.checkbox(
            "Outlines", self.show_outlines
        )
        imgui.same_line()
        imgui.set_next_item_width(110)
        opacity_changed, self.opacity = imgui.slider_float(
            "Opacity", self.opacity, 0.05, 1.0, "%.2f"
        )
        if masks_changed or outlines_changed or opacity_changed:
            self.refresh_overlay()

        imgui.text_disabled(self.status)
        imgui.separator()

        if self.n_rois:
            if draw_progress(self.classes.labels):
                if self.order.next_unlabeled():
                    self.select_roi(self.order.current)
            self.new_label, changed = draw_label_editor(
                self.classes, self.new_label, "_roi"
            )
            picked = draw_label_buttons(self.classes, "_roi")
            if picked is not None:
                self.label_selected(picked)
            if changed:
                self.order.rebuild()
                self.refresh_overlay()

            imgui.separator()
            draw_filter_row(self.order, self.classes, "_roi")
            with imgui_ctx.begin_child("##roi_table", imgui.ImVec2(0, 0)):
                self.scroll_to_selection = draw_roi_table(
                    self.order,
                    self.classes,
                    COLUMNS,
                    {"area": lambda i: f"{self.masks.counts[i]}"},
                    self.scroll_to_selection,
                    table_id="manual_rois",
                    on_select=self.select_roi,
                )
        else:
            imgui.text_disabled("no ROIs yet")

        self.keybinds_open = draw_keybinds_popup(KEYBINDS, self.keybinds_open, "ROI keys")
