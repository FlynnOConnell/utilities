"""Manual ROI drawing + labeling widget for the viewer.

Toggled from ``Widgets > Manual ROI Labeling`` in the preview GUI (``mbo
<path> --widget manualroi`` opens with it on). Laid out like masknmf's
``ClassificationVis``: a panel across the top of the figure carries every
control - drawing tools, background and overlay controls, labeling
progress, the class label editor and buttons - and a "ROIs" tab in the
right-hand widget (see ``widgets/tabs.py``) holds the filter row and the
sortable ROI table. The panel, table, label set, stroke capture and overlay
compositing are the shared widgets from ``masknmf.visualization.imgui``,
so the two GUIs look and behave the same.

Arm "Add ROI", drag a closed stroke around a cell, release and the enclosed
pixels become a mask. ROIs live in a ``RoiLabelStore``: one ``(Z, Y, X)``
uint16 label volume (0 is background, ROI ``i`` is ``i + 1``) so they can
never overlap - pixels already claimed by another ROI are dropped from a
new stroke. A stroke lands on the z-plane the viewer currently shows; T
and C are ignored (masks are shared across time and channels). Data
without a z slider degrades to a single plane.

Each ROI can carry a class label from a user-defined label set (colored
buttons in the panel; keys 1-9 assign, 0 clears) and a free-text note.
The colour mode picks whether ROIs render in their class colour, their own
hue, or the class colour where labelled and their own hue otherwise.

Annotations autosave next to the data as an OME-NGFF-style labels zarr
(``manual_labels.zarr``, see ``mbo_utilities.annotation``) and are restored
from it on relaunch. "Save" writes the full store explicitly.

The background is the viewer's own image graphic, so the bg source combo
picks between the live movie and mean / max / std projections of the plane
on screen. Those same images (plus the movie and the ROI outlines) open in
"Full FOV", the shared ``SummaryImageViewer``.

Each row of the ROI table carries a Run button (extract or demix that ROI
through ``roi_workflow``, writing ``rois_<tag>/`` beside the data), a
Quick trace and an Extract trace button. Traces run on their own thread,
tracked as a job in the process manager, and land in the floating Traces
panel, whose cursor is the viewer's own ``t`` - drag it to scrub the movie.
With drawing off, clicking a background pixel traces that pixel.

With drawing off, clicking an ROI selects it: it is redrawn at
``SELECTED_OPACITY`` behind a white rim, and its row in the table is
highlighted and scrolled into view. Clicking the background clears the
selection. Selecting a listed ROI that lives on another plane jumps the z
slider to it.

Only the first subplot is drawable. Multi-ROI raw ScanImage data opens one
subplot per ROI and the rest are left alone.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui, imgui_ctx, icons_fontawesome_6 as fa
from masknmf.visualization.imgui import (
    OUTLINE_PLACEMENT,
    OUTLINE_PLACEMENTS,
    OUTLINE_WIDTH,
    SELECTED_ALPHA,
    UNLABEL_ALL,
    AsyncLoad,
    LabelSet,
    RoiOrder,
    RowAction,
    StrokeDrawer,
    SummaryImageViewer,
    draw_filter_row,
    draw_keybinds_popup,
    draw_label_buttons,
    draw_label_editor,
    draw_progress,
    draw_roi_table,
    label_image_rgba,
    outline_labels,
)

from mbo_utilities import log
from mbo_utilities.annotation import (
    ROI_COLORS,
    UNLABELED,
    LabelsZarr,
    RoiLabelStore,
    class_color,
)
from mbo_utilities.arrays.features import find_slider_name
from mbo_utilities.gui._imgui_helpers import (
    button_width,
    draw_toolbar_row,
    fit_edge_window,
    fit_width,
    selected_button_style,
    set_tooltip,
)
from mbo_utilities.gui.widgets.trace_panel import TracePanel
from mbo_utilities.gui.widgets.widget_toggles import sub_enabled
from mbo_utilities.roi_workflow import (
    OUT_PREFIX,
    PlaneMovie,
    demix_rois,
    extract_rois,
    pixel_trace,
    roi_trace,
)

__all__ = [
    "COLOR_MODES",
    "MAX_OUTLINE_WIDTH",
    "OUTLINE_PLACEMENTS",
    "PANEL_HEIGHT",
    "PANEL_LOCATION",
    "PROCESSES",
    "PROJECTIONS",
    "ManualRoiWidget",
    "SAVE_NAME",
    "SELECTED_OPACITY",
    "attach_roi_widget",
    "compute_projections",
    "detach_roi_widget",
    "labels_path",
    "roi_widgets_available",
]

# the top panel: the control rows, like ClassificationVis. Rows wrap on
# overflow and the window grows to fit (fit_edge_window), so this is a floor
PANEL_LOCATION = "top"
PANEL_HEIGHT = 110
TOOLBAR_HEIGHT = PANEL_HEIGHT

# below these widths the panel / tab collapse to a placeholder line rather
# than clip: the tools need one button row, the table its three columns
MIN_PANEL_WIDTH = 260
MIN_TAB_WIDTH = 150

# a stroke enclosing fewer unclaimed pixels than this is a misclick, not a cell
MIN_ROI_PIXELS = 9

# fill opacity of the selected ROI, so it pops out of the others whatever the
# global opacity is set to
SELECTED_OPACITY = SELECTED_ALPHA

SAVE_NAME = "manual_labels.zarr"

DEFAULT_LABEL_NAMES = ("cell", "not cell")

COLUMNS = ("id", "label", "area")

# what the per-row Run button and "run in view" send the ROIs through
PROCESSES = ("extract", "demix")

# how ROI fills and outlines are coloured
COLOR_MODES = ("label + roi", "label", "roi")
UNLABELED_COLOR = (0.62, 0.62, 0.62)

# a thicker boundary than this swallows a small ROI whole
MAX_OUTLINE_WIDTH = 6


def _icon(name: str, fallback: str) -> str:
    """FontAwesome glyph by constant name, or a plain-text stand-in."""
    return getattr(fa, name, "") or fallback


# the per-row action buttons: glyph only, so the column stays narrow, with
# the name in the tooltip
RUN_ICON = _icon("ICON_FA_PLAY", ">")
QUICK_TRACE_ICON = _icon("ICON_FA_BOLT", "~")
EXTRACT_TRACE_ICON = _icon("ICON_FA_FLASK", "F")

# background sources besides the live movie. Reduced over an evenly spaced
# sample rather than every frame: this runs off a lazy array and only has to
# be good enough to draw ROIs on.
PROJECTIONS = {
    "mean": lambda stack: stack.mean(axis=0),
    "max": lambda stack: stack.max(axis=0),
    "std": lambda stack: stack.std(axis=0),
}
PROJECTION_FRAMES = 300

KEYBINDS = (
    ("a", "arm / disarm drawing"),
    ("esc", "stop drawing"),
    ("ctrl+z", "undo last ROI"),
    ("delete", "delete the selected ROI"),
    ("up / down", "previous / next ROI"),
    ("shift + up / down", "jump 10 ROIs"),
    ("left / right", "previous / next label group"),
    ("1-9", "assign label to the selected ROI"),
    ("0", "clear its label"),
    ("u", "jump to next unlabeled ROI"),
    ("m", "toggle masks"),
    ("o", "toggle outlines"),
    ("b", "toggle background"),
    ("click", "select the ROI under the cursor (drawing off)"),
)

_CODE_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)
_LOADING_COLOR = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)
_ERROR_COLOR = imgui.ImVec4(1.0, 0.4, 0.3, 1.0)


def labels_path(fpath) -> Path:
    """Where ``fpath``'s annotations live: ``manual_labels.zarr`` beside a
    file, or inside a directory."""
    base = Path.cwd() if fpath is None else Path(fpath)
    return (base.parent if base.suffix else base) / SAVE_NAME


def roi_widgets_available() -> bool:
    """True when masknmf's shared imgui widgets import (they are what the
    ROI panel and table are built from)."""
    try:
        import masknmf.visualization.imgui  # noqa: F401
    except Exception:
        return False
    return True


def _byte_exact(rgb) -> np.ndarray:
    """uint8 rgb as 0-1 floats that come back as the same bytes after the
    overlay's ``(colors * 255).astype(uint8)`` truncation."""
    return (np.asarray(rgb, np.float32) + 0.5) / 255.0


def compute_projections(movie, n_frames: int = PROJECTION_FRAMES) -> dict:
    """Reduce an evenly spaced sample of ``movie`` down to one image each."""
    total = int(movie.shape[0])
    picks = np.unique(np.linspace(0, total - 1, min(total, n_frames)).astype(int))
    stack = np.stack([np.asarray(movie[int(i)], dtype=np.float32) for i in picks])
    return {name: reduce(stack) for name, reduce in PROJECTIONS.items()}


class _PlaneOrder(RoiOrder):
    """``RoiOrder`` with an extra "only this z-plane" filter."""

    def __init__(self, columns, labels, n_items):
        super().__init__(columns, labels, n_items)
        self.plane: int | None = None
        self.planes = np.zeros(0, np.int64)

    def rebuild(self):
        super().rebuild()
        if self.plane is None or not len(self.order):
            return
        current = self.current
        self.order = self.order[self.planes[self.order] == self.plane]
        hits = np.flatnonzero(self.order == current)
        self.pos = int(hits[0]) if len(hits) else int(
            min(self.pos, max(len(self.order) - 1, 0))
        )


class ManualRoiWidget:
    """Freehand ROI painting + class labeling on top of a ``MboNDViewer``.

    Parameters
    ----------
    iw : MboNDViewer
        The viewer to draw on. Only its first subplot is drawable.
    fpath : path-like, optional
        The data path; annotations autosave to ``manual_labels.zarr`` beside
        it and are restored from there on construction.
    label_names : iterable of str
        Class labels to seed the label set with (existing names are kept).
    store : RoiLabelStore, optional
        Adopt this in-memory store instead of starting empty / restoring
        from disk - how the widget keeps its ROIs across an off/on toggle
        when there is nothing on disk to restore from.
    """

    def __init__(self, iw, fpath=None, label_names=(), store=None):
        self.iw = iw
        self.figure = iw.figure
        self.fpath = Path(fpath) if fpath is not None else None
        self.logger = log.get("gui.manual_roi")
        # one-shot: the host selects the ROIs tab on the next frame
        self.focus_tab = False

        self.subplot = iw.figure[0, 0]
        self.image = iw.graphics[0]
        self.ny, self.nx = self.image.data.value.shape[:2]

        # z axis of the viewer, resolved through the same aliases the rest of
        # the GUI uses ("z", "Zplane", "Z-plane", "Cube-slice", ...); None
        # when the data has no depth
        self.zdim = find_slider_name(iw.dim_names, "z")
        self.tdim = find_slider_name(iw.dim_names, "t")
        nz = 1
        if self.zdim is not None:
            rr = iw.ndwidget.indices.ref_ranges.get(self.zdim)
            if rr is not None:
                nz = max(int(rr.stop - rr.start), 1)

        if store is not None and (store.nz, store.ny, store.nx) == (nz, self.ny, self.nx):
            self.store = store
        else:
            self.store = RoiLabelStore(nz, self.ny, self.nx, min_pixels=MIN_ROI_PIXELS)
        for name in label_names:
            self.store.add_label_name(name)

        self.selected = -1
        self.status = "press Add ROI to start"
        self._save_error: str | None = None
        self._writer: LabelsZarr | None = None
        self.new_label = ""
        self._note_buf = ""
        self.keybinds_open = False
        self.scroll_to_selection = False

        # overlay appearance
        self.show_masks = True
        self.show_outlines = True
        self.opacity = 0.45
        # px per side of the mask boundary; masknmf's shared default, kept
        # here so it can be tuned per host
        self.outline_width = OUTLINE_WIDTH
        self.outline_alpha = 1.0
        self.placement_idx = OUTLINE_PLACEMENTS.index(OUTLINE_PLACEMENT)
        self.color_mode_idx = 0

        # background: the viewer's own graphic, showing either the live movie
        # or one of the projections of the plane on screen
        self.show_bg = True
        self.bg_alpha = 1.0
        self.bg_source_idx = 0
        self._projections: dict[str, np.ndarray] = {}
        # (z, c) the cached projections were reduced from
        self._projection_key: tuple[int, int] | None = None
        self._loader = AsyncLoad()
        self._live_frame: np.ndarray | None = None
        self._frozen_index: dict | None = None
        self._contours: list | None = None

        # ui-side mirrors of the store: the masknmf label set + table order
        self.classes = LabelSet(0, self.store.label_names)
        self.order = _PlaneOrder({"area": np.zeros(0, np.int64)}, self.classes.labels, 0)
        self.order.set_range_column("area")

        self.z = self._current_z()
        if self.zdim is not None:
            iw.ndwidget.indices.add_event_handler(self._on_indices)

        self.overlay = self.subplot.add_image(
            np.zeros((self.ny, self.nx, 4), np.uint8),
            name="manual_roi_overlay",
            alpha_mode="blend",
            offset=(0, 0, 1),
        )
        # the overlay is literal uint8 RGBA, not data to be contrast-stretched.
        # fastplotlib auto-ranges it off the initial all-zero array, giving
        # vmin == vmax == 0, which saturates every non-zero channel to 255 -
        # tab10 class colours all come out white, and only hues with a zero
        # channel survive. Pin the full byte range instead.
        self.overlay.vmin, self.overlay.vmax = 0, 255
        # keep the overlay out of picking: the tooltip then reports the image
        # intensity under the cursor as it does without this widget, and ROI
        # hit-testing is a label lookup here rather than a pick
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False

        self.drawer = StrokeDrawer(self.subplot, self.add_roi, self._pick)

        self.summary = SummaryImageViewer(
            iw.figure,
            title="Full FOV",
            roi_provider=self.roi_contours,
            on_export=self.export_image,
        )

        # traces (floating panel, built on first use) and pipeline runs, both
        # computed off the draw thread through the movie contract
        # ``arr[t, c, z, y, x]`` so any lazy array the viewer shows works.
        # Every trace click is its own thread and process-manager job; the
        # results come back through a queue drained on the draw loop.
        self.traces: TracePanel | None = None
        self.pixel_traces = True
        self._trace_results: queue.Queue = queue.Queue()
        self._trace_threads: list[threading.Thread] = []
        self.process = PROCESSES[0]
        self.run_status = ""
        self._run_job = AsyncLoad()

        # keep the window handle so the panel can grow when its rows wrap,
        # and so teardown never reclaims an edge another widget took over
        self.tools_window = iw.figure.add_imgui_window(
            ImguiWindow(update_call=self.draw_panel),
            location=PANEL_LOCATION,
            size=PANEL_HEIGHT,
            title="ROIs",
        )

        self._closed = False
        self._restore()
        self._resync()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Take everything back off the figure: panel, overlays, handlers,
        and hand the viewer's graphic back to the live movie."""
        if self._closed:
            return
        self._closed = True
        self.set_drawing(False)
        try:
            self.set_bg_source(0)
            self.show_bg = True
            self.bg_alpha = 1.0
            self.apply_background()
        except Exception:
            self.logger.debug("background restore failed", exc_info=True)
        renderer = self.subplot.renderer
        for fn, kind in (
            (self.drawer._down, "pointer_down"),
            (self.drawer._move, "pointer_move"),
            (self.drawer._up, "pointer_up"),
        ):
            try:
                renderer.remove_event_handler(fn, kind)
            except (KeyError, ValueError):
                pass
        if self.zdim is not None:
            try:
                self.iw.ndwidget.indices.remove_event_handler(self._on_indices)
            except (KeyError, ValueError, AttributeError):
                pass
        for graphic in (self.overlay, self.drawer.line):
            try:
                self.subplot.delete_graphic(graphic)
            except (KeyError, ValueError):
                pass
        try:
            self.summary.close()
            self.summary.cleanup()
        except Exception:
            self.logger.debug("summary viewer cleanup failed", exc_info=True)
        if self.traces is not None:
            self.traces.close()
            self.traces = None
        if self.figure.imgui_windows.get(PANEL_LOCATION) is self.tools_window:
            self.figure.remove_imgui_window(PANEL_LOCATION)
        self.tools_window = None

    # ------------------------------------------------------------------
    # z / t tracking
    # ------------------------------------------------------------------

    def _current_z(self) -> int:
        if self.zdim is None:
            return 0
        return int(np.clip(self.iw.indices[self.zdim], 0, self.store.nz - 1))

    def _on_indices(self, _indices):
        # fires on every slider move, including t during playback - only a
        # real z change costs anything
        z = self._current_z()
        if z == self.z:
            return
        self.z = z
        self._contours = None
        if self.drawer.stroke:
            # a stroke cannot span planes; drop one interrupted by a z jump
            self.drawer.stroke = []
            self.drawer.line.visible = False
        if self.order.plane is not None:
            self.order.plane = z
            self.order.rebuild()
        self.refresh_overlay()

    def current_frame(self) -> int:
        """The viewer's t (0 for data without a time axis)."""
        if self.tdim is None:
            return 0
        return int(self.iw.indices[self.tdim])

    def set_frame(self, frame: int):
        """Move the viewer's t, so dragging the trace cursor scrubs the movie."""
        if self.tdim is None:
            return
        movie = self.movie()
        limit = (int(movie.shape[0]) - 1) if movie is not None else 0
        self.iw.indices[self.tdim] = int(np.clip(frame, 0, limit))

    # ------------------------------------------------------------------
    # store <-> ui mirrors
    # ------------------------------------------------------------------

    @property
    def labels(self) -> np.ndarray:
        """the current z-plane's label image (a view into the store volume)"""
        return self.store.labels[self.z]

    @property
    def counts(self) -> list[int]:
        """per-ROI pixel counts, all planes, in ROI order"""
        return self.store.counts

    @property
    def n_rois(self) -> int:
        return len(self.store.rois)

    @property
    def drawing(self) -> bool:
        return self.drawer.armed

    @property
    def stroke(self) -> list:
        return self.drawer.stroke

    @property
    def stroke_line(self):
        return self.drawer.line

    def _resync(self):
        """Rebuild the label set and table order from the store."""
        rois = self.store.rois
        self.classes = LabelSet(
            len(rois),
            self.store.label_names,
            np.array([r.class_index for r in rois], np.int64),
        )
        # a restored store can name classes past the ones it lists
        self.store.label_names = self.classes.names
        planes = np.array([r.z for r in rois], np.int64)
        columns = {"area": np.asarray(self.store.counts, np.int64)}
        if self.store.nz > 1:
            columns["z"] = planes
        self.order.columns = columns
        self.order.labels = self.classes.labels
        self.order.n_items = len(rois)
        self.order.planes = planes
        self.order.set_range_column("area")
        self.order.rebuild()
        self._contours = None

    def _sync_store_from_classes(self):
        """Push label-set edits (add / remove a class) back into the store."""
        self.store.label_names = tuple(self.classes.names)
        for record, ci in zip(self.store.rois, self.classes.labels):
            record.class_index = int(ci)

    @property
    def columns(self) -> tuple[str, ...]:
        return COLUMNS + (("z",) if self.store.nz > 1 else ())

    def _formatters(self) -> dict:
        return {
            "area": lambda i: f"{self.store.rois[i].area}",
            "z": lambda i: f"{self.store.rois[i].z + 1}",
        }

    # ------------------------------------------------------------------
    # mask state
    # ------------------------------------------------------------------

    def set_drawing(self, on: bool):
        """Arm or disarm stroke drawing.

        Left-drag pans by default, so the pan binding is lifted for as long
        as drawing is armed; wheel zoom and right-drag zoom stay live.
        """
        if on == self.drawer.armed:
            return
        self.drawer.arm(on)
        self.status = (
            "drag a closed stroke around a cell" if on else f"{self.n_rois} ROIs"
        )

    def add_roi(self, stroke):
        """Fill a closed stroke and store it as the next label on plane z."""
        if len(stroke) < 3:
            self.status = "stroke too short"
            return
        points = np.round(np.asarray(stroke, np.float32)).astype(np.int32)
        points[:, 0] = points[:, 0].clip(0, self.nx - 1)
        points[:, 1] = points[:, 1].clip(0, self.ny - 1)

        filled = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(filled, [points], 1)
        index = self.store.add_roi(self.z, filled.astype(bool))
        if index is None:
            self.status = f"under {MIN_ROI_PIXELS} free px, not added"
            return
        self._resync()
        self.select_roi(index)
        self._autosave()

    def _pick(self, row: int, col: int):
        index = self.store.roi_at(self.z, row, col)
        self.select_roi(index)
        if index < 0 and self.pixel_traces:
            self.trace_pixel(row, col)

    pick_roi = _pick

    def select_roi(self, index: int | None):
        """Select ROI ``index``; anything out of range clears the selection.

        The selected ROI is redrawn at ``SELECTED_OPACITY`` behind a white
        rim, and the table scrolls its row into view on the next frame.
        Selecting an ROI on another plane jumps the z slider to it.
        """
        self.selected = index if index is not None and 0 <= index < self.n_rois else -1
        self.scroll_to_selection = True
        if self.selected < 0:
            self._note_buf = ""
            self.status = f"{self.n_rois} ROIs"
        else:
            record = self.store.rois[self.selected]
            self._note_buf = record.note
            self.order.goto(self.selected)
            self.status = f"ROI {self.selected}: {record.area} px"
            if self.zdim is not None and record.z != self.z:
                # fires _on_indices, which refreshes the overlay for that plane
                self.iw.indices[self.zdim] = record.z
        self.summary.set_highlight(self.selected_bbox())
        self.refresh_overlay()

    def selected_bbox(self) -> tuple | None:
        """``(y0, x0, h, w)`` of the selected ROI, for the Full FOV highlight."""
        if self.selected < 0:
            return None
        record = self.store.rois[self.selected]
        rows, cols = np.nonzero(self.store.labels[record.z] == self.selected + 1)
        if not rows.size:
            return None
        y0, x0 = int(rows.min()), int(cols.min())
        return y0, x0, int(rows.max()) - y0 + 1, int(cols.max()) - x0 + 1

    def roi_contours(self) -> list:
        """Outline of every ROI on the current plane as ``(N, 2)`` y/x
        points, for the Full FOV."""
        if self._contours is not None:
            return self._contours
        contours = []
        plane = self.labels
        for i in self.store.rois_on_plane(self.z):
            mask = (plane == i + 1).astype(np.uint8)
            found, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours.extend(c.reshape(-1, 2)[:, ::-1] for c in found if len(c) > 1)
        self._contours = contours
        return contours

    def step(self, delta: int):
        """Move through the filtered/sorted view by ``delta`` ROIs."""
        if self.order.step(delta):
            self.select_roi(self.order.current)

    def step_group(self, direction: int):
        """Jump to the first ROI in view of the next/previous label class."""
        if self.order.step_group(direction):
            self.select_roi(self.order.current)

    def next_unlabeled(self):
        """Jump to the next unlabeled ROI in the current view."""
        if self.order.next_unlabeled():
            self.select_roi(self.order.current)

    def delete_roi(self, index: int):
        """Drop one ROI and renumber the labels above it."""
        if not self.store.delete_roi(index):
            return
        # deleting renumbers every ROI above it, so traces named by index no
        # longer belong to the ROIs that key them
        if self.traces is not None:
            self.traces.clear()
        self._resync()
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index}"
        self._autosave()

    def clear(self):
        self.store.clear()
        if self.traces is not None:
            self.traces.clear()
        self._resync()
        self.select_roi(-1)
        self.status = "cleared"
        self._autosave()

    def assign_class(self, class_index: int):
        """Give the selected ROI a class label; UNLABELED (-1) clears it."""
        if self.selected < 0:
            return
        self.store.set_class(self.selected, class_index)
        # the store may have been edited directly (label names added), so
        # rebuild the mirrors rather than patch them
        self._resync()
        self.order.goto(self.selected)
        self.status = f"ROI {self.selected}: {self.classes.name_of(self.selected)}"
        self.refresh_overlay()
        self._autosave()

    label_selected = assign_class

    def unlabel_all(self):
        for i in range(self.n_rois):
            self.store.set_class(i, UNLABELED)
        self.classes.assign(range(self.n_rois), UNLABELED)
        self.order.rebuild()
        self.refresh_overlay()
        self.status = f"cleared {self.n_rois} labels"
        self._autosave()

    # ------------------------------------------------------------------
    # overlay
    # ------------------------------------------------------------------

    @property
    def color_mode(self) -> str:
        return COLOR_MODES[self.color_mode_idx]

    @property
    def outline_placement(self) -> str:
        return OUTLINE_PLACEMENTS[self.placement_idx]

    def edges(self, width: int | None = None, placement: str | None = None) -> np.ndarray:
        """Outline of the current plane's labels, as a label image."""
        return outline_labels(
            self.labels,
            self.outline_width if width is None else width,
            self.outline_placement if placement is None else placement,
        )

    def _colors(self) -> np.ndarray:
        """One RGB (0-1) per ROI, per the colour mode.

        ``label`` gives every ROI in a class the same colour and greys the
        unlabeled; ``roi`` gives each its own hue so neighbours stay apart
        while drawing; ``label + roi`` does the first where a label exists
        and the second where it does not.
        """
        mode = self.color_mode
        colors = np.zeros((max(self.n_rois, 1), 3), np.float32)
        for i, record in enumerate(self.store.rois):
            identity = _byte_exact(ROI_COLORS[i % len(ROI_COLORS)])
            if mode == "roi":
                colors[i] = identity
            elif record.class_index >= 0:
                colors[i] = _byte_exact(
                    [round(c * 255) for c in class_color(record.class_index)]
                )
            else:
                colors[i] = identity if mode == "label + roi" else UNLABELED_COLOR
        return colors

    def refresh_overlay(self):
        """Recompose the RGBA overlay from the current plane's labels."""
        self.overlay.visible = self.show_masks or self.show_outlines
        if not self.overlay.visible:
            return
        self.overlay.data = label_image_rgba(
            self.labels,
            colors=self._colors(),
            alpha=self.opacity,
            selected=self.selected,
            show_masks=self.show_masks,
            show_outlines=self.show_outlines,
            edges=self.edges(),
            outline_width=self.outline_width,
            outline_alpha=self.outline_alpha,
            outline_placement=self.outline_placement,
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _save_target(self) -> Path:
        return labels_path(self.fpath)

    def _output_path(self, name: str) -> Path:
        """``name`` beside the data (the labels zarr's directory)."""
        return self._save_target().parent / name

    def _restore(self):
        """Adopt a previously saved labels zarr next to the data, if any."""
        if self.fpath is None:
            return
        target = self._save_target()
        self._writer = LabelsZarr(target)
        if not target.exists():
            return
        try:
            store = LabelsZarr.load(target)
        except (OSError, ValueError) as e:
            self.logger.warning(f"could not restore {target}: {e}")
            self.status = f"restore failed: {e}"
            return
        if (store.nz, store.ny, store.nx) != (
            self.store.nz,
            self.store.ny,
            self.store.nx,
        ):
            self.logger.warning(
                f"{target} is {store.labels.shape}, data wants "
                f"{self.store.labels.shape}; starting fresh"
            )
            self.status = "saved labels do not match this data, starting fresh"
            return
        for name in self.store.label_names:
            store.add_label_name(name)
        store.min_pixels = MIN_ROI_PIXELS
        self.store = store
        self.status = f"restored {len(store.rois)} ROIs"
        self.refresh_overlay()

    def _autosave(self):
        if self._writer is None:
            return
        try:
            self._writer.save_dirty(self.store, source_path=self.fpath)
            self._save_error = None
        except OSError as e:
            self._save_error = f"autosave failed: {e}"

    def save(self):
        """Write the full store to ``manual_labels.zarr`` next to the data."""
        target = self._save_target()
        if self._writer is None or self._writer.path != target:
            self._writer = LabelsZarr(target)
        try:
            self._writer.save(self.store, source_path=self.fpath)
        except OSError as e:
            self._save_error = f"save failed: {e}"
            return
        self._save_error = None
        self.status = f"saved to {target.name}"
        self.logger.info(f"saved {self.n_rois} ROIs to {target}")

    # ------------------------------------------------------------------
    # the movie contract
    # ------------------------------------------------------------------

    def _channel(self) -> int:
        cdim = find_slider_name(self.iw.dim_names, "c")
        return int(self.iw.indices[cdim]) if cdim is not None else 0

    def movie(self, z: int | None = None) -> PlaneMovie | None:
        """``(T, Y, X)`` view of the viewer's array on plane ``z`` (default:
        the plane on screen) and the channel on screen, or None when the
        array is not something the movie contract can wrap."""
        try:
            arr = self.iw.data[0]
        except (AttributeError, IndexError, TypeError):
            return None
        z = self.z if z is None else int(z)
        try:
            nz = PlaneMovie(arr, z=0, c=0).nz
            return PlaneMovie(arr, z=(z if nz > 1 else 0), c=self._channel())
        except (ValueError, IndexError, TypeError):
            return None

    plane_movie = movie

    # ------------------------------------------------------------------
    # background and summary images
    # ------------------------------------------------------------------

    @property
    def bg_sources(self) -> list[str]:
        """The live movie, then the projections. Reduced on first selection."""
        return ["movie", *PROJECTIONS]

    def _plane_key(self) -> tuple[int, int]:
        return self.z, self._channel()

    def drop_stale_projections(self):
        """Forget projections reduced from a plane the viewer has left."""
        if self._projection_key is None:
            return
        if self._plane_key() != self._projection_key:
            self._projections = {}
            self._projection_key = None

    def request_projections(self):
        """Kick off the projection reduce for the plane on screen, once."""
        self.drop_stale_projections()
        movie = self.movie()
        if movie is None or self._loader.busy or self._projections:
            return
        self._projection_key = self._plane_key()
        self._loader.start(
            lambda: compute_projections(movie), "computing projections..."
        )

    def _poll_projections(self):
        result = self._loader.poll()
        if result is None:
            return
        self._projections = result
        name = self.bg_sources[self.bg_source_idx]
        if name in result:
            self._freeze_background(name)
        self.refresh_full_fov()

    def apply_background(self):
        self.image.visible = self.show_bg
        self.image.alpha = self.bg_alpha

    def _freeze_background(self, name: str):
        """Park the projection in the viewer's graphic, keeping the live frame."""
        if self._frozen_index is None:
            self._live_frame = np.asarray(self.image.data.value).copy()
        self.image.data = self._projections[name]
        self._frozen_index = dict(self.iw.current_index)

    def set_bg_source(self, index: int):
        """Freeze the viewer graphic to a projection, or hand it back."""
        self.drop_stale_projections()
        sources = self.bg_sources
        self.bg_source_idx = int(np.clip(index, 0, len(sources) - 1))
        name = sources[self.bg_source_idx]
        if name == "movie":
            if self._frozen_index is not None and self._live_frame is not None:
                self.image.data = self._live_frame
            self._frozen_index = None
        elif name in self._projections:
            self._freeze_background(name)
        else:
            # the reduce runs on a thread; _poll_projections parks it
            self.request_projections()

    def _follow_viewer(self):
        """Snap back to "movie" once the viewer has repainted the graphic.

        The NDWidget rewrites the image whenever a slider moves, so a frozen
        projection is gone the moment the user scrubs; the combo has to say
        so. The projections themselves stay cached - they only depend on
        the plane / channel, which ``drop_stale_projections`` checks.
        """
        if self._frozen_index is None:
            return
        if dict(self.iw.current_index) != self._frozen_index:
            self._frozen_index = None
            self.bg_source_idx = 0

    def open_full_fov(self):
        """Open the shared summary viewer over the movie, the projections and
        the current plane's ROI outlines."""
        self.request_projections()
        movie = self.movie()
        # movies first: set_images resolves `selected` against both sets
        self.summary.set_movies({"movie": movie} if movie is not None else {})
        self.summary.set_images(
            {"current frame": np.asarray(self.image.data.value), **self._projections},
            selected=self.bg_sources[self.bg_source_idx],
        )
        self.summary.set_highlight(self.selected_bbox())
        self.summary.open()

    def refresh_full_fov(self):
        """Push newly computed projections into an already-open Full FOV."""
        if self.summary.is_open and self._projections:
            self.summary.set_images(
                {
                    "current frame": np.asarray(self.image.data.value),
                    **self._projections,
                }
            )

    def export_image(self, key: str, array):
        """Write one summary image beside the data, as the Full FOV's export."""
        out = self._output_path(f"summary_{key.replace(' ', '_')}.tif")
        try:
            tifffile.imwrite(out, np.asarray(array, dtype=np.float32))
        except OSError as error:
            self.status = f"export failed: {error}"
            return
        self.status = f"exported {out.name}"
        self.logger.info(f"exported {key} to {out}")

    # ------------------------------------------------------------------
    # traces: each click is a thread + process-manager job, landing in the
    # floating Traces panel
    # ------------------------------------------------------------------

    def _trace_panel(self) -> TracePanel:
        if self.traces is None:
            self.traces = TracePanel(self.iw.figure)
            self.traces.on_scrub = self.set_frame
        self.traces.show()
        return self.traces

    @property
    def trace_busy(self) -> bool:
        self._trace_threads = [t for t in self._trace_threads if t.is_alive()]
        return bool(self._trace_threads)

    def trace_extractor(self):
        """The pipeline "Extract trace" would run, or None."""
        from mbo_utilities.gui.widgets.pipelines import get_trace_extractors

        extractors = get_trace_extractors()
        return extractors[0] if extractors else None

    def trace_disabled(self, index: int) -> str | None:
        """Why the trace actions cannot run for ``index``, or None."""
        movie = self.movie()
        if movie is None or int(movie.shape[0]) < 2:
            return "no (T, Y, X) movie behind this view"
        return None

    def extract_disabled(self, index: int) -> str | None:
        reason = self.trace_disabled(index)
        if reason is not None:
            return reason
        if self.trace_extractor() is None:
            return (
                "no installed pipeline can extract traces.\n"
                "Install one, e.g. uv pip install mbo_utilities[all]"
            )
        return None

    def _start_trace(self, description: str, work):
        """Run ``work`` on a thread, tracked as a process-manager job.

        Every click gets its own job and its own thread: the button is the
        only feedback the user has otherwise, and two ROIs can be tracing
        at once. ``work`` returns ``{trace name: 1-D array}``; the results
        come back through a queue and are added to the Traces panel by
        :meth:`_poll_jobs` on the draw loop.
        """
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        job = get_process_manager().start_job("roi_trace", description)
        self.status = f"{description} started"
        self._trace_panel().status = f"{description}..."

        def run():
            try:
                result = work()
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"{description} failed")
                job.fail(f"{type(error).__name__}: {error}")
                self._trace_results.put((description, None, str(error)))
                return
            if not result:
                job.fail("the extractor returned nothing")
                self._trace_results.put((description, None, "no trace returned"))
                return
            frames = int(np.asarray(next(iter(result.values()))).reshape(-1).size)
            job.done(f"{frames} frames")
            self._trace_results.put((description, result, None))

        thread = threading.Thread(target=run, name=f"roi-trace", daemon=True)
        self._trace_threads.append(thread)
        thread.start()

    def trace_pixel(self, row: int, col: int):
        """Plot ``arr[:, c, z, row, col]`` in the trace panel."""
        movie = self.movie()
        if movie is None:
            return
        name = f"px ({row}, {col}) z{self.z + 1}"
        self._start_trace(
            f"pixel trace - ({row}, {col})",
            lambda: {name: pixel_trace(movie, row, col)},
        )

    def quick_trace(self, index: int):
        """Mean of the ROI's pixels per frame - no pipeline, just the mask."""
        if not 0 <= index < self.n_rois:
            return
        record = self.store.rois[index]
        movie = self.movie(record.z)
        if movie is None:
            return
        mask = self.store.labels[record.z] == (index + 1)
        name = f"ROI {index}"
        self._start_trace(
            f"quick trace - ROI {index}",
            lambda: {name: roi_trace(movie, mask)},
        )

    def extract_trace(self, index: int):
        """Hand the mask to a pipeline's own extractor (suite2p, say)."""
        if not 0 <= index < self.n_rois:
            return
        pipeline = self.trace_extractor()
        record = self.store.rois[index]
        movie = self.movie(record.z)
        if movie is None or pipeline is None:
            return
        # one ROI at a time: the pipeline sees only this mask, so its
        # neuropil ring is not carved up by the others
        labels = (self.store.labels[record.z] == (index + 1)).astype(np.uint16)

        def work():
            result = pipeline.extract_traces(movie, labels)
            if not result:
                return None
            return {
                f"ROI {index} {key} ({pipeline.name})": np.asarray(value).reshape(-1)
                for key, value in result.items()
            }

        self._start_trace(f"{pipeline.name} trace - ROI {index}", work)

    def _poll_traces(self):
        """Drain finished trace jobs; called once per frame from the panel."""
        while True:
            try:
                description, result, error = self._trace_results.get_nowait()
            except queue.Empty:
                return
            if error is not None:
                self.status = f"{description} failed: {error}"
                if self.traces is not None:
                    self.traces.status = self.status
                continue
            panel = self._trace_panel()
            for name, y in result.items():
                panel.add(name, y)
            self.status = f"{description}: {panel.status}"

    _poll_trace = _poll_traces

    # ------------------------------------------------------------------
    # pipeline runs through roi_workflow
    # ------------------------------------------------------------------

    def _run_out_dir(self, tag: str) -> Path | None:
        if self.fpath is None:
            return None
        return labels_path(self.fpath).parent / f"{OUT_PREFIX}{tag}"

    def run_rois(self, indices: list[int], tag: str):
        """Send ``indices`` through ``self.process`` (extract | demix) on the
        viewer's own array, writing ``rois_<tag>/`` beside the data."""
        indices = [i for i in indices if 0 <= i < self.n_rois]
        if not indices:
            self.run_status = "nothing to run"
            return
        if self._run_job.busy:
            self.run_status = "a run is still going"
            return
        out_dir = self._run_out_dir(tag)
        if out_dir is None:
            self.run_status = "no data path to write beside"
            return
        arr = self.iw.data[0]
        store = self.store
        process = self.process
        c = self._channel()
        planes = sorted({store.rois[i].z for i in indices})
        logger = self.logger

        def _job():
            outs = []
            for z in planes:
                on_plane = [i for i in indices if store.rois[i].z == z]
                dest = out_dir if len(planes) == 1 else out_dir / f"z{z + 1:02d}"
                fn = extract_rois if process == "extract" else demix_rois
                out = fn(arr, store, on_plane, z=z, c=c, out_dir=dest, tag=tag, logger=logger)
                if out is not None:
                    outs.append(out)
            return outs

        self.run_status = f"{process}: {len(indices)} ROI(s) -> {out_dir.name}..."
        self._run_job.start(_job, status=self.run_status)

    def run_roi(self, index: int):
        self.run_rois([index], f"roi{index:04d}")

    def run_in_view(self):
        self.run_rois([int(i) for i in self.order.order], "manual")

    def _poll_jobs(self):
        self._poll_traces()
        outs = self._run_job.poll()
        if outs is not None:
            self.run_status = (
                f"done: {', '.join(Path(o).name for o in outs)}" if outs else "done: nothing written"
            )
            self.logger.info(f"roi run {self.run_status}")
        elif self._run_job.error:
            self.run_status = f"run failed: {self._run_job.error}"
            self._run_job.error = None

    # ------------------------------------------------------------------
    # keys
    # ------------------------------------------------------------------

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
        if self.selected >= 0 and imgui.is_key_pressed(imgui.Key.delete, False):
            self.delete_roi(self.selected)
        if imgui.is_key_pressed(imgui.Key.b, False):
            self.show_bg = not self.show_bg
            self.apply_background()
        if imgui.is_key_pressed(imgui.Key.m, False):
            self.show_masks = not self.show_masks
            self.refresh_overlay()
        if imgui.is_key_pressed(imgui.Key.o, False):
            self.show_outlines = not self.show_outlines
            self.refresh_overlay()
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.next_unlabeled()
        stride = 10 if io.key_shift else 1
        if imgui.is_key_pressed(imgui.Key.up_arrow):
            self.step(-stride)
        if imgui.is_key_pressed(imgui.Key.down_arrow):
            self.step(stride)
        if imgui.is_key_pressed(imgui.Key.left_arrow):
            self.step_group(-1)
        if imgui.is_key_pressed(imgui.Key.right_arrow):
            self.step_group(1)
        if self.selected >= 0:
            picked = self.classes.hotkey_pressed()
            if picked is not None:
                self.assign_class(picked)

    # ------------------------------------------------------------------
    # imgui: top panel
    # ------------------------------------------------------------------

    def draw_panel(self):
        """Top edge window: tools, background, overlay, labeling - the same
        rows as ``ClassificationVis``, each gated by its Widgets-menu
        subwidget toggle.

        Rows wrap on overflow and the window grows to fit, so nothing runs
        off the right edge however narrow the figure or long the label set.
        Narrower than ``MIN_PANEL_WIDTH`` the panel collapses to one line.
        """
        self._poll_projections()
        self._poll_jobs()
        self.drop_stale_projections()
        self._follow_viewer()
        self.handle_keys()
        with fit_width("ROI tools", min_width=MIN_PANEL_WIDTH) as shown:
            if shown:
                self._draw_tool_rows()
        self.summary.draw()
        if self.traces is not None:
            self.traces.frame_marker = self.current_frame() if self.tdim else None
            self.traces.draw()
        self.keybinds_open = draw_keybinds_popup(KEYBINDS, self.keybinds_open, "ROI keys")
        if self.tools_window is not None:
            fit_edge_window(self.tools_window, PANEL_HEIGHT)

    def _draw_tool_rows(self):
        if sub_enabled("manual_roi", "tools"):
            self._draw_tools_row()
        if sub_enabled("manual_roi", "overlay"):
            self._draw_background_row()
            self._draw_mask_row()
        else:
            draw_toolbar_row([(None, imgui.calc_text_size(self.status).x, self._draw_status)])
        if sub_enabled("manual_roi", "labels"):
            self._draw_labels_row()

    def _draw_tools_row(self):
        def _add():
            with selected_button_style(self.drawer.armed):
                if imgui.button("Add ROI"):
                    self.set_drawing(not self.drawer.armed)
            set_tooltip(
                "Drag a closed stroke around a cell, release to fill it. "
                "'a' toggles, esc stops, ctrl+Z undoes. With drawing off, "
                "click an ROI to select it; keys 1-9 label the selection, 0 "
                "unlabels it.",
                show_mark=False,
            )

        def _undo():
            if imgui.button("Undo"):
                self.delete_roi(self.n_rois - 1)

        def _clear():
            if imgui.button("Clear"):
                self.clear()

        def _save():
            if imgui.button("Save"):
                self.save()

        def _prev():
            if imgui.button("prev"):
                self.step(-1)

        def _next():
            if imgui.button("next"):
                self.step(1)

        def _pos():
            changed, pos = imgui.slider_int(
                "##pos", self.order.pos, 0, max(len(self.order.order) - 1, 0)
            )
            if changed and len(self.order.order):
                self.order.pos = pos
                self.select_roi(self.order.current)

        def _fov():
            if imgui.button("Open full FOV"):
                self.open_full_fov()

        def _keybinds():
            if imgui.button("keybinds"):
                self.keybinds_open = True

        draw_toolbar_row(
            [
                (None, button_width("Add ROI"), _add),
                (None, button_width("Undo"), _undo),
                (None, button_width("Clear"), _clear),
                (None, button_width("Save"), _save),
                (None, button_width("prev"), _prev),
                (None, button_width("next"), _next),
                ("roi", 120.0, _pos),
                (None, button_width("Open full FOV"), _fov),
                (None, button_width("keybinds"), _keybinds),
                (None, imgui.calc_text_size("Autosaved").x, self._draw_save_note),
            ]
        )

    def _draw_background_row(self):
        box = imgui.get_frame_height()

        def _bg():
            changed, self.show_bg = imgui.checkbox("bg image", self.show_bg)
            imgui.same_line(0, 4)
            imgui.text_disabled("(b)")
            if changed:
                self.apply_background()

        def _source():
            changed, index = imgui.combo("##bg-source", self.bg_source_idx, self.bg_sources)
            if changed:
                self.set_bg_source(index)
            set_tooltip(
                "The viewer's own image. Projections are reduced from an "
                "evenly spaced sample of the plane on screen; moving a "
                "slider hands the image back to the movie.",
                show_mark=False,
            )

        def _bg_alpha():
            changed, self.bg_alpha = imgui.slider_float(
                "##bg-alpha", self.bg_alpha, 0.0, 1.0, "%.2f"
            )
            if changed:
                self.apply_background()

        def _px_trace():
            _, self.pixel_traces = imgui.checkbox("px trace", self.pixel_traces)
            set_tooltip(
                "With drawing off, clicking a background pixel plots its "
                "trace over time in the Traces panel.",
                show_mark=False,
            )

        def _traces():
            if imgui.button(f"{fa.ICON_FA_CHART_LINE} Traces"):
                if self.traces is not None and self.traces.visible:
                    self.traces.hide()
                else:
                    self._trace_panel()

        draw_toolbar_row(
            [
                (None, button_width("bg image (b)") + box, _bg),
                (None, 110.0, _source),
                ("bg opacity", 80.0, _bg_alpha),
                (None, button_width("px trace") + box, _px_trace),
                (None, button_width(f"{fa.ICON_FA_CHART_LINE} Traces"), _traces),
                (None, imgui.calc_text_size(self._status_text()).x, self._draw_status),
            ]
        )

    def _draw_mask_row(self):
        box = imgui.get_frame_height()
        dirty = []

        def _masks():
            changed, self.show_masks = imgui.checkbox("masks", self.show_masks)
            imgui.same_line(0, 4)
            imgui.text_disabled("(m)")
            dirty.append(changed)

        def _fill():
            changed, self.opacity = imgui.slider_float(
                "##fill", self.opacity, 0.05, 1.0, "%.2f"
            )
            set_tooltip("Opacity of the mask interiors.", show_mark=False)
            dirty.append(changed)

        def _outlines():
            changed, self.show_outlines = imgui.checkbox("outlines", self.show_outlines)
            imgui.same_line(0, 4)
            imgui.text_disabled("(o)")
            dirty.append(changed)

        def _width():
            changed, self.outline_width = imgui.slider_int(
                "##width", self.outline_width, 1, MAX_OUTLINE_WIDTH
            )
            set_tooltip(
                "Outline thickness. The overlay is one texel per data pixel, "
                "so 1 px is the thinnest a line can be drawn.",
                show_mark=False,
            )
            dirty.append(changed)

        def _placement():
            changed, self.placement_idx = imgui.combo(
                "##placement", self.placement_idx, list(OUTLINE_PLACEMENTS)
            )
            set_tooltip(
                "Which side of the boundary the outline sits on.\n"
                "outer:  on the background just outside the mask - costs no\n"
                "        mask pixel, so 1-3 px structures keep every pixel\n"
                "inner:  on the mask's own outermost pixels\n"
                "center: straddles the boundary, eating a pixel each way",
                show_mark=False,
            )
            dirty.append(changed)

        def _line():
            changed, self.outline_alpha = imgui.slider_float(
                "##line", self.outline_alpha, 0.05, 1.0, "%.2f"
            )
            set_tooltip("Opacity of the outlines, independent of the fill.", show_mark=False)
            dirty.append(changed)

        def _color():
            changed, self.color_mode_idx = imgui.combo(
                "##color-by", self.color_mode_idx, list(COLOR_MODES)
            )
            set_tooltip(
                "label:       one colour per class, unlabeled grey\n"
                "label + roi: class colour where labelled, a unique hue otherwise\n"
                "roi:         a unique hue per ROI",
                show_mark=False,
            )
            dirty.append(changed)

        draw_toolbar_row(
            [
                (None, button_width("masks (m)") + box, _masks),
                ("fill", 70.0, _fill),
                (None, button_width("outlines (o)") + box, _outlines),
                ("px", 70.0, _width),
                (None, 80.0, _placement),
                ("line", 70.0, _line),
                ("color by", 100.0, _color),
            ]
        )
        if any(dirty):
            self.refresh_overlay()

    def _status_text(self) -> str:
        if self._loader.busy:
            return self._loader.status
        return self._loader.error or self._save_error or self.status

    def _draw_status(self):
        if self._loader.busy:
            imgui.text_colored(_LOADING_COLOR, self._loader.status)
        elif self._loader.error or self._save_error:
            imgui.text_colored(_ERROR_COLOR, self._loader.error or self._save_error)
        else:
            imgui.text_disabled(self.status)

    def _draw_labels_row(self):
        # progress, then the class editor and buttons. Those masknmf widgets
        # lay themselves out with same_line, so the row is given a new line
        # when the editor would not fit after the progress
        if self.n_rois:
            if draw_progress(self.classes.labels):
                self.next_unlabeled()
            right = imgui.get_cursor_screen_pos().x + imgui.get_content_region_avail().x
            if imgui.get_item_rect_max().x + 24 + 125 + button_width("add") + button_width("del") <= right:
                imgui.same_line(0, 24)
        self.new_label, changed = draw_label_editor(self.classes, self.new_label, "_roi")
        if changed:
            self._sync_store_from_classes()
            self.order.rebuild()
            self.refresh_overlay()
            self._autosave()
        picked = draw_label_buttons(self.classes, "_roi")
        if picked == UNLABEL_ALL:
            self.unlabel_all()
        elif picked is not None:
            self.assign_class(picked)

    def _draw_save_note(self):
        if self._writer is None:
            imgui.text_disabled("autosave off - ROIs are kept in memory only")
            if imgui.is_item_hovered():
                imgui.set_tooltip("open a file to autosave beside it, or press Save")
            return
        imgui.text_disabled("Autosaved")
        if not imgui.is_item_hovered():
            return
        imgui.begin_tooltip()
        imgui.text(f"labels zarr: {self._save_target()}")
        imgui.text_colored(
            _CODE_COLOR,
            "from mbo_utilities.annotation import LabelsZarr\n"
            "\n"
            f'store = LabelsZarr.load(r"{self._save_target()}")\n'
            "store.labels        # (Z, Y, X) uint16; 0 = bg, ROI i = i + 1\n"
            "store.rois          # per-ROI plane, area, class index, note\n"
            "store.label_names   # class names; index = class value",
        )
        imgui.end_tooltip()

    # ------------------------------------------------------------------
    # imgui: right tab
    # ------------------------------------------------------------------

    def draw_tab(self):
        """The ROIs tab body: filter row, the ROI table, the selected note,
        and the run row."""
        if self.store.nz > 1:
            on = self.order.plane is not None
            changed, on = imgui.checkbox(f"this plane (z {self.z + 1}/{self.store.nz})", on)
            if changed:
                self.order.plane = self.z if on else None
                self.order.rebuild()
        draw_filter_row(self.order, self.classes, "_roi")

        footer = 3 * imgui.get_frame_height_with_spacing() + 12
        with imgui_ctx.begin_child("##roi_table", imgui.ImVec2(0, -footer)):
            if self.n_rois:
                pos = self.order.pos
                self.scroll_to_selection = draw_roi_table(
                    self.order,
                    self.classes,
                    self.columns,
                    self._formatters(),
                    self.scroll_to_selection,
                    table_id="manual_rois",
                    on_select=self.select_roi,
                    actions=self.row_actions,
                )
                if self.order.pos != pos and self.order.current is not None:
                    self.select_roi(self.order.current)
            else:
                imgui.text_disabled("no ROIs yet")

        imgui.separator()
        if self.selected >= 0:
            imgui.set_next_item_width(-1)
            changed, self._note_buf = imgui.input_text_with_hint(
                "##note", "note", self._note_buf
            )
            if changed:
                self.store.set_note(self.selected, self._note_buf)
            if imgui.is_item_deactivated_after_edit():
                self._autosave()
        else:
            imgui.text_disabled("select an ROI to note")
        if imgui.button("Delete selected"):
            self.delete_roi(self.selected)
        # run row: what the row buttons / "in view" send ROIs through
        imgui.set_next_item_width(90)
        changed, sel = imgui.combo("##process", PROCESSES.index(self.process), list(PROCESSES))
        if changed:
            self.process = PROCESSES[sel]
        set_tooltip(
            "extract: suite2p-style traces from the drawn masks.\n"
            "demix: masknmf NMF seeded with the drawn masks.\n"
            "Outputs land in rois_<tag>/ beside the data.",
            show_mark=False,
        )
        imgui.same_line()
        if imgui.button(f"{RUN_ICON} in view"):
            self.run_in_view()
        set_tooltip("Run every ROI currently listed", show_mark=False)
        if self.run_status:
            imgui.same_line()
            imgui.text_disabled(self.run_status)

    draw_table = draw_tab

    @property
    def row_actions(self) -> tuple[RowAction, ...]:
        """The icon-only buttons at the end of each table row."""
        return (
            RowAction(RUN_ICON, f"Run - {self.process} this ROI", self.run_roi),
            RowAction(
                QUICK_TRACE_ICON,
                "Quick trace - average this ROI's pixels over time",
                self.quick_trace,
                self.trace_disabled,
            ),
            RowAction(
                EXTRACT_TRACE_ICON,
                self._extract_tooltip(),
                self.extract_trace,
                self.extract_disabled,
            ),
        )

    def _row_actions(self) -> tuple[RowAction, ...]:
        return self.row_actions

    def _extract_tooltip(self) -> str:
        pipeline = self.trace_extractor()
        which = pipeline.name if pipeline is not None else "a pipeline"
        return f"Extract trace - run this ROI's mask through {which}"


# ----------------------------------------------------------------------
# host wiring: PreviewDataWidget owns at most one widget as ``manual_roi``
# (see PreviewDataWidget.sync_manual_roi, driven by the Widgets-menu toggle)
# ----------------------------------------------------------------------


def attach_roi_widget(parent: Any, focus: bool = False) -> ManualRoiWidget | None:
    """Turn the ROI widget on for a ``PreviewDataWidget``.

    Adds the top panel and overlays, and makes the "ROIs" tab appear in the
    right widget. ROIs from an earlier toggle in this session are adopted
    when nothing newer is on disk. Returns the widget, or None (logged) when
    it could not be built.
    """
    widget = getattr(parent, "manual_roi", None)
    if widget is not None:
        widget.focus_tab = widget.focus_tab or focus
        return widget
    fpath = parent.fpath[0] if isinstance(parent.fpath, list) else parent.fpath
    try:
        widget = ManualRoiWidget(
            parent.image_widget,
            fpath,
            label_names=DEFAULT_LABEL_NAMES,
            store=getattr(parent, "_manual_roi_store", None),
        )
    except Exception:
        parent.logger.warning("manual ROI widget unavailable", exc_info=True)
        parent.manual_roi = None
        return None
    widget.focus_tab = focus
    parent.manual_roi = widget
    return widget


def detach_roi_widget(parent: Any) -> None:
    """Turn the ROI widget off, keeping its store for the next toggle."""
    widget = getattr(parent, "manual_roi", None)
    if widget is None:
        return
    parent._manual_roi_store = widget.store
    widget.close()
    parent.manual_roi = None
