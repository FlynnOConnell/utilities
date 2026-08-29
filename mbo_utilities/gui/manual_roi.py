"""Manual ROI drawing + labeling widget for the viewer.

Toggled from the ``Widgets > ROIs`` menu of the preview GUI (``mbo <path>
--widget manualroi`` opens with it on). Laid out like masknmf's
``ClassificationVis``: a panel across the top of the figure carries every
control - drawing tools, overlay toggles, labeling progress, the class
label editor and buttons - and a "ROIs" tab in the right-hand widget holds
the filter row and the sortable ROI table. The panel, table, label set,
stroke capture and overlay compositing are the shared widgets from
``masknmf.visualization.imgui``, so the two GUIs look and behave the same.

Arm "Add ROI", drag a closed stroke around a cell, release and the enclosed
pixels become a mask. ROIs live in a ``RoiLabelStore``: one ``(Z, Y, X)``
uint16 label volume (0 is background, ROI ``i`` is ``i + 1``) so they can
never overlap - pixels already claimed by another ROI are dropped from a
new stroke. A stroke lands on the z-plane the viewer currently shows; T
and C are ignored (masks are shared across time and channels). Data
without a z slider degrades to a single plane.

Each ROI can carry a class label from a user-defined label set (colored
buttons in the panel; keys 1-9 assign, 0 clears) and a free-text note.
Classified ROIs render in their class color, unclassified ones keep their
own hue.

Annotations autosave next to the data as an OME-NGFF-style labels zarr
(``manual_labels.zarr``, see ``mbo_utilities.annotation``) and are restored
from it on relaunch. "Save" writes the full store explicitly.

With drawing off, clicking an ROI selects it: it is redrawn at
``SELECTED_OPACITY`` behind a white rim, and its row in the table is
highlighted and scrolled into view. Clicking the background clears the
selection. Selecting a listed ROI that lives on another plane jumps the z
slider to it.

Only the first subplot is drawable. Multi-ROI raw ScanImage data opens one
subplot per ROI and the rest are left alone.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui, imgui_ctx, icons_fontawesome_6 as fa
from masknmf.visualization.imgui import (
    AsyncLoad,
    LabelSet,
    RoiOrder,
    StrokeDrawer,
    SummaryImageViewer,
    draw_filter_row,
    draw_keybinds_popup,
    draw_label_buttons,
    draw_label_editor,
    draw_progress,
    draw_roi_table,
    label_edges,
    label_image_rgba,
)
from masknmf.visualization.imgui.overlay import SELECTED_ALPHA

from mbo_utilities import log
from mbo_utilities.annotation import UNLABELED, LabelsZarr, RoiLabelStore
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
from mbo_utilities.roi_workflow import (
    OUT_PREFIX,
    PlaneMovie,
    demix_rois,
    extract_rois,
    pixel_trace,
    roi_trace,
)

# what the per-row Run button and "run in view" send the ROIs through
PROCESSES = ("extract", "demix")

__all__ = [
    "ManualRoiWidget",
    "SAVE_NAME",
    "SELECTED_OPACITY",
    "attach_roi_widget",
    "detach_roi_widget",
    "labels_path",
    "roi_widgets_available",
]

# the top panel: three control rows, like ClassificationVis
TOOLBAR_HEIGHT = 110

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

KEYBINDS = (
    ("a", "arm / disarm drawing"),
    ("esc", "stop drawing"),
    ("ctrl+z", "undo last ROI"),
    ("delete", "delete the selected ROI"),
    ("1-9", "assign label to the selected ROI"),
    ("0", "clear its label"),
    ("click", "select the ROI under the cursor (drawing off)"),
)

_DANGER_COLORS = (
    imgui.ImVec4(0.75, 0.15, 0.15, 0.8),
    imgui.ImVec4(0.90, 0.20, 0.20, 1.0),
)


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
        self.fpath = Path(fpath) if fpath is not None else None
        self.logger = log.get("gui.manual_roi")
        # one-shot: the host selects the ROIs tab on the next frame
        self.focus_tab = False

        self.subplot = iw.figure[0, 0]
        self.ny, self.nx = iw.graphics[0].data.value.shape[:2]

        # z axis of the viewer, resolved through the same aliases the rest of
        # the GUI uses ("z", "Zplane", "Z-plane", "Cube-slice", ...); None
        # when the data has no depth
        self.zdim = find_slider_name(iw.dim_names, "z")
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

        self.show_masks = True
        self.show_outlines = True
        self.opacity = 0.45

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
        # keep the overlay out of picking: the tooltip then reports the image
        # intensity under the cursor as it does without this widget, and ROI
        # hit-testing is a label lookup here rather than a pick
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False

        self.drawer = StrokeDrawer(self.subplot, self.add_roi, self._pick)

        self.summary = SummaryImageViewer(iw.figure, title="Full FOV")

        # quick traces (bottom panel, built on first use) and pipeline runs,
        # both computed off the draw thread through the movie contract
        # ``arr[t, c, z, y, x]`` so any lazy array the viewer shows works
        self.traces: TracePanel | None = None
        self.pixel_traces = True
        self.process = PROCESSES[0]
        self.run_status = ""
        self._trace_job = AsyncLoad()
        self._run_job = AsyncLoad()

        # keep the window handle so the panel can grow when its rows wrap
        self.tools_window = iw.figure.add_imgui_window(
            ImguiWindow(update_call=self.draw_panel),
            location="top",
            size=TOOLBAR_HEIGHT,
            title="ROIs",
        )

        self._closed = False
        self._restore()
        self._resync()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Take everything back off the figure: panel, overlays, handlers."""
        if self._closed:
            return
        self._closed = True
        self.set_drawing(False)
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
            self.summary.cleanup()
        except Exception:
            self.logger.debug("summary viewer cleanup failed", exc_info=True)
        if self.traces is not None:
            self.traces.close()
            self.traces = None
        if self.iw.figure.imgui_windows.get("top") is self.tools_window:
            self.iw.figure.remove_imgui_window("top")
        self.tools_window = None

    # ------------------------------------------------------------------
    # z tracking
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
        if self.drawer.stroke:
            # a stroke cannot span planes; drop one interrupted by a z jump
            self.drawer.stroke = []
            self.drawer.line.visible = False
        if self.order.plane is not None:
            self.order.plane = z
            self.order.rebuild()
        self.refresh_overlay()

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

    def select_roi(self, index: int):
        """Select ROI ``index``; anything out of range clears the selection.

        The selected ROI is redrawn at ``SELECTED_OPACITY`` behind a white
        rim, and the table scrolls its row into view on the next frame.
        Selecting an ROI on another plane jumps the z slider to it.
        """
        self.selected = index if 0 <= index < self.n_rois else -1
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
        self.refresh_overlay()

    def delete_roi(self, index: int):
        """Drop one ROI and renumber the labels above it."""
        if not self.store.delete_roi(index):
            return
        self._resync()
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index}"
        self._autosave()

    def clear(self):
        self.store.clear()
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
        self._autosave()

    def _colors(self) -> np.ndarray:
        """(n, 3) float rgb per ROI: class color where labeled, own hue else."""
        n = max(self.n_rois, 1)
        colors = np.zeros((n, 3), np.float32)
        for i in range(self.n_rois):
            colors[i] = np.asarray(self.store.roi_rgb(i), np.float32) / 255.0
        return colors

    def refresh_overlay(self):
        """Recompose the RGBA overlay from the current plane's labels."""
        self.overlay.visible = self.show_masks or self.show_outlines
        if not self.overlay.visible:
            return
        plane = self.store.labels[self.z]
        self.overlay.data = label_image_rgba(
            plane,
            colors=self._colors(),
            alpha=self.opacity,
            selected=self.selected,
            show_masks=self.show_masks,
            show_outlines=self.show_outlines,
            edges=label_edges(plane),
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _save_target(self) -> Path:
        return labels_path(self.fpath)

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
    # full-FOV popup
    # ------------------------------------------------------------------

    def open_full_fov(self):
        """masknmf's summary-image popup over the frame on screen and the
        current plane's label image."""
        frame = np.asarray(self.iw.graphics[0].data.value, np.float32)
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1)
        images = {"current frame": frame}
        if self.n_rois:
            images["ROI labels"] = self.store.labels[self.z].astype(np.float32)
        self.summary.set_images(images, selected="current frame")
        self.summary.open()

    # ------------------------------------------------------------------
    # quick traces and pipeline runs, through the movie contract
    # ------------------------------------------------------------------

    def _channel(self) -> int:
        cdim = find_slider_name(self.iw.dim_names, "c")
        return int(self.iw.indices[cdim]) if cdim is not None else 0

    def movie(self, z: int | None = None) -> PlaneMovie:
        """``(T, Y, X)`` view of the viewer's array on plane ``z`` (default:
        the plane on screen) and the channel on screen."""
        arr = self.iw.data[0]
        z = self.z if z is None else int(z)
        nz = PlaneMovie(arr, z=0, c=0).nz
        return PlaneMovie(arr, z=(z if nz > 1 else 0), c=self._channel())

    def _trace_panel(self) -> TracePanel:
        if self.traces is None:
            self.traces = TracePanel(self.iw.figure)
        self.traces.show()
        return self.traces

    def trace_pixel(self, row: int, col: int):
        """Plot ``arr[:, c, z, row, col]`` in the trace panel."""
        if self._trace_job.busy:
            self.status = "trace still computing"
            return
        movie = self.movie()
        name = f"px ({row}, {col}) z{self.z + 1}"
        self._trace_job.start(
            lambda: (name, pixel_trace(movie, row, col)), status=f"tracing {name}"
        )
        self._trace_panel().status = f"tracing {name}..."

    def quick_trace(self, index: int):
        """Plot the mean over ROI ``index`` per frame in the trace panel."""
        if not 0 <= index < self.n_rois:
            return
        if self._trace_job.busy:
            self.status = "trace still computing"
            return
        record = self.store.rois[index]
        mask = self.store.labels[record.z] == (index + 1)
        movie = self.movie(record.z)
        name = f"ROI {index}"
        self._trace_job.start(lambda: (name, roi_trace(movie, mask)), status=f"tracing {name}")
        self._trace_panel().status = f"tracing {name}..."

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
        got = self._trace_job.poll()
        if got is not None:
            name, y = got
            self._trace_panel().add(name, y)
            self.status = f"{name}: {y.size} frames"
        elif self._trace_job.error:
            self.status = f"trace failed: {self._trace_job.error}"
            self._trace_job.error = None
            if self.traces is not None:
                self.traces.status = self.status
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
        if self.selected >= 0:
            picked = self.classes.hotkey_pressed()
            if picked is not None:
                self.assign_class(picked)

    # ------------------------------------------------------------------
    # imgui: top panel
    # ------------------------------------------------------------------

    def draw_panel(self):
        """Top edge window: tools, overlay controls, labeling - the same rows
        as ``ClassificationVis``.

        Rows wrap on overflow and the window grows to fit, so nothing runs
        off the right edge however narrow the figure or long the label set.
        Narrower than ``MIN_PANEL_WIDTH`` the panel collapses to one line.
        """
        self.handle_keys()
        self._poll_jobs()
        with fit_width("ROI tools", min_width=MIN_PANEL_WIDTH) as shown:
            if shown:
                self._draw_tool_rows()
        self.summary.draw()
        if self.traces is not None:
            self.traces.draw()
        self.keybinds_open = draw_keybinds_popup(KEYBINDS, self.keybinds_open, "ROI keys")
        if self.tools_window is not None:
            fit_edge_window(self.tools_window, TOOLBAR_HEIGHT)

    def _draw_tool_rows(self):
        box = imgui.get_frame_height()

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

        def _fov():
            if imgui.button("Open full FOV"):
                self.open_full_fov()

        def _keybinds():
            if imgui.button("keybinds"):
                self.keybinds_open = True

        def _masks():
            changed, self.show_masks = imgui.checkbox("masks", self.show_masks)
            if changed:
                self.refresh_overlay()

        def _outlines():
            changed, self.show_outlines = imgui.checkbox("outlines", self.show_outlines)
            if changed:
                self.refresh_overlay()

        def _opacity():
            changed, self.opacity = imgui.slider_float(
                "##opacity", self.opacity, 0.05, 1.0, "%.2f"
            )
            if changed:
                self.refresh_overlay()

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

        def _status():
            imgui.text_disabled(self.status)
            if self._save_error is not None:
                imgui.same_line(0, 10)
                imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.3, 1.0), self._save_error)

        status = self.status + (self._save_error or "")
        draw_toolbar_row(
            [
                (None, button_width("Add ROI"), _add),
                (None, button_width("Undo"), _undo),
                (None, button_width("Clear"), _clear),
                (None, button_width("Save"), _save),
                (None, button_width("Open full FOV"), _fov),
                (None, button_width("keybinds"), _keybinds),
                (None, imgui.calc_text_size("Autosaved").x, self._draw_save_note),
            ]
        )
        draw_toolbar_row(
            [
                (None, button_width("masks") + box, _masks),
                (None, button_width("outlines") + box, _outlines),
                ("roi opacity", 110.0, _opacity),
                (None, button_width("px trace") + box, _px_trace),
                (None, button_width(f"{fa.ICON_FA_CHART_LINE} Traces"), _traces),
                (None, imgui.calc_text_size(status).x, _status),
            ]
        )

        # labeling row: progress, then the class editor and buttons. Those
        # masknmf widgets lay themselves out with same_line, so the row is
        # given a new line when the editor would not fit after the progress
        if draw_progress(self.classes.labels):
            if self.order.next_unlabeled():
                self.select_roi(self.order.current)
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
        if picked is not None:
            self.assign_class(picked)
        if self.classes.names and self.n_rois:
            imgui.same_line(0, 10)
            imgui.push_style_color(imgui.Col_.button, _DANGER_COLORS[0])
            imgui.push_style_color(imgui.Col_.button_hovered, _DANGER_COLORS[1])
            if imgui.button("unlabel all"):
                self.unlabel_all()
            imgui.pop_style_color(2)

    def _draw_save_note(self):
        if self._writer is None:
            imgui.text_disabled("autosave off - ROIs are kept in memory only")
            if imgui.is_item_hovered():
                imgui.set_tooltip("open a file to autosave beside it, or press Save")
            return
        imgui.text_disabled("Autosaved")
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"labels zarr: {self._save_target()}")

    # ------------------------------------------------------------------
    # imgui: right tab
    # ------------------------------------------------------------------

    def draw_tab(self):
        """The ROIs tab body: filter row, the ROI table, the selected note."""
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
        if imgui.button(f"{fa.ICON_FA_PLAY} in view"):
            self.run_in_view()
        set_tooltip("Run every ROI currently listed", show_mark=False)
        if self.run_status:
            imgui.same_line()
            imgui.text_disabled(self.run_status)

    @property
    def row_actions(self) -> tuple:
        """The icon-only buttons at the end of each table row."""
        return (
            (fa.ICON_FA_PLAY, f"{self.process} this ROI", self.run_roi),
            (fa.ICON_FA_CHART_LINE, "quick trace", self.quick_trace),
        )


# ----------------------------------------------------------------------
# host wiring: PreviewDataWidget owns at most one widget as ``manual_roi``
# ----------------------------------------------------------------------


def attach_roi_widget(parent, focus: bool = False) -> ManualRoiWidget | None:
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


def detach_roi_widget(parent) -> None:
    """Turn the ROI widget off, keeping its store for the next toggle."""
    widget = getattr(parent, "manual_roi", None)
    if widget is None:
        return
    parent._manual_roi_store = widget.store
    widget.close()
    parent.manual_roi = None
