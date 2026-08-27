"""Manual ROI drawing GUI.

Opened with ``mbo <path> --widget manualroi``. Independent of
PreviewDataWidget: it hangs two imgui windows off the viewer figure - tools
across the top, the ROI list down the right - and paints masks onto the
image the way the cellpose GUI does. Arm "Add ROI", drag a closed stroke
around a cell, release and the enclosed pixels become a mask.

ROIs live in one uint16 label image (0 is background, ROI ``i`` is ``i + 1``)
so they can never overlap: pixels already claimed by another ROI are dropped
from a new stroke. "Save" writes that label image next to the data as
``manual_masks.npy``.

With drawing off, clicking an ROI selects it: it is redrawn at
``SELECTED_OPACITY`` behind a white rim, and its row in the ROI list is
highlighted and scrolled into view. Clicking the background clears the
selection.

Only the first subplot is drawable. Multi-ROI raw ScanImage data opens one
subplot per ROI and the rest are left alone.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from imgui_bundle import imgui, imgui_ctx

from mbo_utilities import log
from mbo_utilities.gui._imgui_helpers import selected_button_style, set_tooltip

__all__ = ["ManualRoiWidget"]

TOOLBAR_HEIGHT = 42
PANEL_WIDTH = 260

# a stroke enclosing fewer unclaimed pixels than this is a misclick, not a cell
MIN_ROI_PIXELS = 9

# fill opacity of the selected ROI, so it pops out of the others whatever the
# global opacity is set to
SELECTED_OPACITY = 0.9

# a press that travels further than this many screen px is a drag, not a pick
CLICK_SLOP = 4

# one fully saturated color per label. hues are shuffled so consecutive ROIs
# contrast, and saturating them keeps the fill readable over gnuplot2, which
# already owns most of the pale end of the spectrum.
_HSV = np.zeros((180, 1, 3), np.uint8)
_HSV[:, 0, 0] = np.random.default_rng(0).permutation(180)
_HSV[:, 0, 1:] = 255
_COLORS = cv2.cvtColor(_HSV, cv2.COLOR_HSV2RGB).reshape(-1, 3)

# 5x5 leaves a 2 px rim, thick enough that toggling outlines is obvious
_RIM = np.ones((5, 5), np.uint8)


class ManualRoiWidget:
    """Freehand ROI painting on top of a ``MboNDViewer``."""

    def __init__(self, iw, fpath=None):
        self.iw = iw
        self.fpath = Path(fpath) if fpath is not None else None
        self.logger = log.get("gui.manual_roi")

        self.subplot = iw.figure[0, 0]
        self.ny, self.nx = iw.graphics[0].data.value.shape[:2]

        self.labels = np.zeros((self.ny, self.nx), np.uint16)
        self.counts: list[int] = []
        self.selected = -1
        self.status = "click Add ROI to start"

        self.drawing = False
        self.stroke: list[tuple[float, float]] = []
        self.pan_control = None
        self.press = None
        self.scroll_to_selection = False

        self.show_masks = True
        self.show_outlines = True
        self.opacity = 0.45

        self.overlay = self.subplot.add_image(
            np.zeros((self.ny, self.nx, 4), np.uint8),
            name="manual_roi_overlay",
            alpha_mode="blend",
            offset=(0, 0, 1),
        )
        self.stroke_line = self.subplot.add_line(
            np.zeros((2, 3), np.float32),
            colors="magenta",
            thickness=2.0,
            name="manual_roi_stroke",
            offset=(0, 0, 2),
            visible=False,
        )

        # keep both overlays out of picking: the tooltip then reports the
        # image intensity under the cursor as it does without this widget,
        # instead of the overlay's rgba, and ROI hit-testing is a label
        # lookup here rather than a pick
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False
        self.stroke_line.world_object.material.pick_write = False

        self.subplot.renderer.add_event_handler(self.on_pointer_down, "pointer_down")
        self.subplot.renderer.add_event_handler(self.on_pointer_move, "pointer_move")
        self.subplot.renderer.add_event_handler(self.on_pointer_up, "pointer_up")

        iw.figure.add_imgui_window(location="top", size=TOOLBAR_HEIGHT)(self.draw_tools)
        iw.figure.add_imgui_window(
            location="right", size=PANEL_WIDTH, title="Manual ROI"
        )(self.draw_rois)

    # ------------------------------------------------------------------
    # mask state
    # ------------------------------------------------------------------

    def set_drawing(self, on: bool):
        """Arm or disarm stroke drawing.

        Left-drag pans by default, so the pan binding is lifted for as long
        as drawing is armed; wheel zoom and right-drag zoom stay live.
        """
        if on == self.drawing:
            return
        self.drawing = on
        controls = self.subplot.controller.controls
        if on:
            self.pan_control = controls.pop("mouse1", None)
            self.status = "drag a closed stroke around a cell"
        else:
            if self.pan_control is not None:
                controls["mouse1"] = self.pan_control
                self.pan_control = None
            self.status = f"{len(self.counts)} ROIs"
        self.stroke = []
        self.stroke_line.visible = False

    def add_roi(self, stroke):
        """Fill a closed stroke and store it as the next label."""
        if len(stroke) < 3:
            self.status = "stroke too short"
            return
        points = np.round(np.asarray(stroke, np.float32)).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self.nx - 1)
        points[:, 1] = points[:, 1].clip(0, self.ny - 1)

        filled = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        rows, cols = np.nonzero(filled.astype(bool) & (self.labels == 0))
        if rows.size < MIN_ROI_PIXELS:
            self.status = f"under {MIN_ROI_PIXELS} free px, not added"
            return

        self.counts.append(int(rows.size))
        self.labels[rows, cols] = len(self.counts)
        self.select_roi(len(self.counts) - 1)

    def select_roi(self, index: int):
        """Select ROI ``index``; anything out of range clears the selection.

        The selected ROI is redrawn at ``SELECTED_OPACITY`` behind a white
        rim, and the ROI list scrolls its row into view on the next frame.
        """
        self.selected = index if 0 <= index < len(self.counts) else -1
        self.scroll_to_selection = True
        if self.selected < 0:
            self.status = f"{len(self.counts)} ROIs"
        else:
            self.status = f"ROI {self.selected + 1}: {self.counts[self.selected]} px"
        self.refresh_overlay()

    def delete_roi(self, index: int):
        """Drop one ROI and renumber the labels above it."""
        if not 0 <= index < len(self.counts):
            return
        self.labels[self.labels == index + 1] = 0
        self.labels[self.labels > index + 1] -= 1
        self.counts.pop(index)
        self.select_roi(min(index, len(self.counts) - 1))
        self.status = f"deleted ROI {index + 1}"

    def clear(self):
        self.labels[:] = 0
        self.counts.clear()
        self.select_roi(-1)
        self.status = "cleared"

    def refresh_overlay(self):
        """Recompose the RGBA overlay from the label image.

        A blended fill washes out over a bright cell whatever color it is
        drawn in, so outlines carry the mask boundaries and the fill only
        tints. The morphological gradient of the *label* image keeps the
        seam between two touching ROIs, which eroding a binary mask loses.
        """
        self.overlay.visible = self.show_masks or self.show_outlines
        if not self.overlay.visible:
            return
        rgba = np.zeros((self.ny, self.nx, 4), np.uint8)
        painted = self.labels > 0
        # all-False when nothing is selected, since label 0 is never painted
        chosen = painted & (self.labels == self.selected + 1)
        if self.show_masks:
            rgba[painted, :3] = _COLORS[(self.labels[painted] - 1) % len(_COLORS)]
            rgba[painted, 3] = int(255 * self.opacity)
            rgba[chosen, 3] = int(255 * SELECTED_OPACITY)
        if self.show_outlines:
            edges = painted & (
                cv2.morphologyEx(self.labels, cv2.MORPH_GRADIENT, _RIM) > 0
            )
            rgba[edges, :3] = _COLORS[(self.labels[edges] - 1) % len(_COLORS)]
            rgba[edges, 3] = 255
        if chosen.any():
            solid = chosen.astype(np.uint8)
            rgba[(solid - cv2.erode(solid, _RIM)).astype(bool)] = 255
        self.overlay.data = rgba

    def save(self):
        base = Path.cwd() if self.fpath is None else self.fpath
        out = (base.parent if base.suffix else base) / "manual_masks.npy"
        np.save(out, self.labels)
        self.status = f"saved to {out.name}"
        self.logger.info(f"saved {len(self.counts)} ROIs to {out}")

    # ------------------------------------------------------------------
    # canvas events
    # ------------------------------------------------------------------

    def on_pointer_down(self, ev):
        if ev.button != 1:
            return
        pos = self.subplot.map_screen_to_world((ev.x, ev.y))
        if pos is None:
            return
        if self.drawing:
            self.stroke = [(float(pos[0]), float(pos[1]))]
        else:
            self.press = (ev.x, ev.y)

    def on_pointer_move(self, ev):
        if not self.stroke:
            return
        pos = self.subplot.map_screen_to_world((ev.x, ev.y), allow_outside=True)
        if pos is None:
            return
        self.stroke.append((float(pos[0]), float(pos[1])))
        vertices = np.zeros((len(self.stroke), 3), np.float32)
        vertices[:, :2] = self.stroke
        self.stroke_line.data = vertices
        self.stroke_line.visible = True

    def on_pointer_up(self, ev):
        if self.stroke:
            stroke, self.stroke = self.stroke, []
            self.stroke_line.visible = False
            self.add_roi(stroke)
            return
        if self.press is None:
            return
        press, self.press = self.press, None
        if abs(ev.x - press[0]) > CLICK_SLOP or abs(ev.y - press[1]) > CLICK_SLOP:
            return  # that was a pan, not a pick
        pos = self.subplot.map_screen_to_world((ev.x, ev.y))
        if pos is None:
            return
        col, row = int(pos[0]), int(pos[1])
        if 0 <= row < self.ny and 0 <= col < self.nx:
            self.select_roi(int(self.labels[row, col]) - 1)

    # ------------------------------------------------------------------
    # imgui windows
    # ------------------------------------------------------------------

    def draw_tools(self):
        """Top edge window: drawing tools and overlay controls."""
        with selected_button_style(self.drawing):
            if imgui.button("Add ROI"):
                self.set_drawing(not self.drawing)
        set_tooltip(
            "Drag a closed stroke around a cell, release to fill it. "
            "Esc stops drawing, ctrl+Z undoes the last ROI. With drawing off, "
            "click an ROI to select it.",
            show_mark=False,
        )

        imgui.same_line()
        if imgui.button("Undo"):
            self.delete_roi(len(self.counts) - 1)

        imgui.same_line()
        if imgui.button("Clear"):
            self.clear()

        imgui.same_line()
        masks_changed, self.show_masks = imgui.checkbox("Masks", self.show_masks)

        imgui.same_line()
        outlines_changed, self.show_outlines = imgui.checkbox(
            "Outlines", self.show_outlines
        )
        if masks_changed or outlines_changed:
            self.refresh_overlay()

        imgui.same_line()
        imgui.set_next_item_width(110)
        changed, self.opacity = imgui.slider_float(
            "Opacity", self.opacity, 0.05, 1.0, "%.2f"
        )
        if changed:
            self.refresh_overlay()

        imgui.same_line()
        imgui.text_disabled(self.status)

        if self.drawing and imgui.is_key_pressed(imgui.Key.escape):
            self.set_drawing(False)
        if imgui.get_io().key_ctrl and imgui.is_key_pressed(imgui.Key.z, False):
            self.delete_roi(len(self.counts) - 1)

    def draw_rois(self):
        """Right edge window: the ROI list."""
        imgui.text(f"{len(self.counts)} ROIs")
        imgui.separator()

        with imgui_ctx.begin_child("##roi_list", imgui.ImVec2(0, -60)):
            for i, count in enumerate(self.counts):
                label = f"ROI {i + 1}##roi{i}"
                clicked, _ = imgui.selectable(label, self.selected == i)
                if self.selected == i and self.scroll_to_selection:
                    imgui.set_scroll_here_y()
                    self.scroll_to_selection = False
                imgui.same_line(imgui.get_window_width() - 60)
                imgui.text_disabled(f"{count} px")
                if clicked:
                    self.select_roi(i)

        imgui.separator()
        if imgui.button("Delete selected"):
            self.delete_roi(self.selected)
        imgui.same_line()
        if imgui.button("Save"):
            self.save()
