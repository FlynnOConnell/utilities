"""Manual ROI drawing + labeling widget for the viewer.

Toggled from ``Widgets > Manual ROI Labeling`` in the preview GUI (``mbo
<path> --widget manualroi`` opens with it on). A panel across the top of the
figure carries the controls - drawing tools, overlay toggles, labeling
progress, the class label editor and buttons - and two tabs in the
right-hand widget (``widgets/tabs.py``) hold the sortable ROI table and the
trace viewer. The panel, table, label set, stroke capture and overlay
compositing are the shared widgets from ``masknmf.visualization.imgui``.

Arm "Add ROI", drag a closed stroke around a cell, release and the enclosed
pixels become a mask. ROIs live in a ``RoiLabelStore``: one ``(Z, Y, X)``
uint16 label volume (0 is background, ROI ``i`` is ``i + 1``) so they can
never overlap. A stroke lands on the z-plane the viewer currently shows; T
and C are ignored. Data without a z slider degrades to a single plane.

Each ROI can carry a class label (keys 1-9 assign, 0 clears) and a free-text
note. Annotations autosave next to the data as an OME-NGFF-style labels zarr
(``manual_labels.zarr``, see ``mbo_utilities.annotation``) and are restored
from it on relaunch.

Traces come from two places and both land in the Traces tab, per ROI: the
row's quick-trace button (mask mean per frame, on its own thread and tracked
as a process-manager job) and the outputs of a run (``rois_<tag>/F.npy``,
``Fneu.npy``, ``roi_indices.npy``), which are read back when the run
finishes. The trace cursor is the viewer's own ``t``; drag it to scrub.

With drawing off, clicking an ROI selects it; clicking the background clears
the selection. Selecting a listed ROI on another plane jumps the z slider.
Only the first subplot is drawable.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastplotlib.ui import ImguiWindow
from imgui_bundle import imgui, imgui_ctx, icons_fontawesome_6 as fa, implot
from masknmf.visualization.imgui import (
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
from mbo_utilities.gui.widgets.widget_toggles import sub_enabled
from mbo_utilities.roi_workflow import (
    OUT_PREFIX,
    PlaneMovie,
    demix_rois,
    extract_rois,
    roi_trace,
)

__all__ = [
    "ManualRoiWidget",
    "PANEL_LOCATION",
    "PROCESSES",
    "SAVE_NAME",
    "SELECTED_OPACITY",
    "attach_roi_widget",
    "detach_roi_widget",
    "labels_path",
    "roi_widgets_available",
]

PANEL_LOCATION = "top"
PANEL_HEIGHT = 110

# below these widths the panel / tabs collapse to a placeholder line
MIN_PANEL_WIDTH = 260
MIN_TAB_WIDTH = 150

MIN_ROI_PIXELS = 9
SELECTED_OPACITY = SELECTED_ALPHA
SAVE_NAME = "manual_labels.zarr"
DEFAULT_LABEL_NAMES = ("cell", "not cell")
COLUMNS = ("id", "label", "area")
PROCESSES = ("extract", "demix")

RUN_ICON = fa.ICON_FA_PLAY
TRACE_ICON = fa.ICON_FA_CHART_LINE

KEYBINDS = (
    ("a", "arm / disarm drawing"),
    ("esc", "stop drawing"),
    ("ctrl+z", "undo last ROI"),
    ("delete", "delete the selected ROI"),
    ("up / down", "previous / next ROI"),
    ("u", "next unlabeled ROI"),
    ("1-9", "assign label to the selected ROI"),
    ("0", "clear its label"),
    ("click", "select the ROI under the cursor (drawing off)"),
)

_CODE_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)
_ERROR_COLOR = imgui.ImVec4(1.0, 0.4, 0.3, 1.0)
_CURSOR_COLOR = imgui.ImVec4(1.0, 0.85, 0.3, 0.9)


def labels_path(fpath) -> Path:
    """``manual_labels.zarr`` beside a file, or inside a directory."""
    base = Path.cwd() if fpath is None else Path(fpath)
    return (base.parent if base.suffix else base) / SAVE_NAME


def roi_widgets_available() -> bool:
    """True when masknmf's shared imgui widgets import."""
    try:
        import masknmf.visualization.imgui  # noqa: F401
    except Exception:
        return False
    return True


def _byte_exact(rgb) -> np.ndarray:
    """uint8 rgb as 0-1 floats that survive the overlay's ``(c * 255).astype(uint8)``."""
    return (np.asarray(rgb, np.float32) + 0.5) / 255.0


def load_run_traces(out_dir) -> dict[int, dict]:
    """Per-ROI ``{"source", "F", "Fneu"}`` from a ``rois_<tag>/`` output."""
    out_dir = Path(out_dir)
    if not (out_dir / "F.npy").exists() or not (out_dir / "roi_indices.npy").exists():
        return {}
    F = np.load(out_dir / "F.npy")
    Fneu = np.load(out_dir / "Fneu.npy") if (out_dir / "Fneu.npy").exists() else None
    indices = np.load(out_dir / "roi_indices.npy").reshape(-1)
    traces = {}
    for row, i in enumerate(indices):
        entry = {"source": out_dir.name, "F": np.asarray(F[row], np.float32)}
        if Fneu is not None:
            entry["Fneu"] = np.asarray(Fneu[row], np.float32)
        traces[int(i)] = entry
    return traces


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
        Class labels to seed the label set with.
    store : RoiLabelStore, optional
        Adopt this in-memory store instead of starting empty / restoring
        from disk (how ROIs survive an off/on toggle).
    """

    def __init__(self, iw, fpath=None, label_names=(), store=None):
        self.iw = iw
        self.figure = iw.figure
        self.fpath = Path(fpath) if fpath is not None else None
        self.logger = log.get("gui.manual_roi")
        self.focus_tab = False
        self.focus_traces = False

        self.subplot = iw.figure[0, 0]
        self.image = iw.graphics[0]
        self.ny, self.nx = self.image.data.value.shape[:2]

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

        self.show_masks = True
        self.show_outlines = True
        self.opacity = 0.45

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
        # literal RGBA bytes: auto-ranging off the all-zero start saturates every colour to white
        self.overlay.vmin, self.overlay.vmax = 0, 255
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False

        self.drawer = StrokeDrawer(self.subplot, self.add_roi, self._pick)
        self.summary = SummaryImageViewer(iw.figure, title="Full FOV")

        # per-ROI traces, keyed by ROI index; see quick_trace and _poll_jobs
        self.traces: dict[int, dict] = {}
        self.trace_roi = -1
        self._trace_results: queue.Queue = queue.Queue()
        self._trace_threads: list[threading.Thread] = []
        self.process = PROCESSES[0]
        self.run_status = ""
        self._run_job = AsyncLoad()

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
            self.summary.close()
            self.summary.cleanup()
        except Exception:
            self.logger.debug("summary viewer cleanup failed", exc_info=True)
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
        z = self._current_z()
        if z == self.z:
            return
        self.z = z
        if self.drawer.stroke:
            self.drawer.stroke = []
            self.drawer.line.visible = False
        if self.order.plane is not None:
            self.order.plane = z
            self.order.rebuild()
        self.refresh_overlay()

    def current_frame(self) -> int:
        return int(self.iw.indices[self.tdim]) if self.tdim is not None else 0

    def set_frame(self, frame: int):
        """Move the viewer's t; the trace cursor drag scrubs the movie with this."""
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
        """Arm or disarm stroke drawing (lifts the pan binding while armed)."""
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
        self.select_roi(self.store.roi_at(self.z, row, col))

    def select_roi(self, index: int | None):
        """Select ROI ``index``; anything out of range clears the selection.

        Selecting an ROI on another plane jumps the z slider to it; one with
        a trace shows it in the Traces tab.
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
                self.iw.indices[self.zdim] = record.z
            if self.selected in self.traces:
                self.trace_roi = self.selected
        self.refresh_overlay()

    def step(self, delta: int):
        if self.order.step(delta):
            self.select_roi(self.order.current)

    def next_unlabeled(self):
        if self.order.next_unlabeled():
            self.select_roi(self.order.current)

    def delete_roi(self, index: int):
        """Drop one ROI and renumber the labels above it."""
        if not self.store.delete_roi(index):
            return
        self.traces.clear()
        self._resync()
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index}"
        self._autosave()

    def clear(self):
        self.store.clear()
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

    def _colors(self) -> np.ndarray:
        """(n, 3) rgb per ROI: class color where labeled, own hue else."""
        colors = np.zeros((max(self.n_rois, 1), 3), np.float32)
        for i in range(self.n_rois):
            colors[i] = _byte_exact(self.store.roi_rgb(i))
        return colors

    def refresh_overlay(self):
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
            edges=outline_labels(self.labels),
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
        if (store.nz, store.ny, store.nx) != (self.store.nz, self.store.ny, self.store.nx):
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

    def open_full_fov(self):
        """masknmf's summary-image popup over the frame on screen and the labels."""
        frame = np.asarray(self.image.data.value, np.float32)
        if frame.ndim == 3:
            frame = frame[..., :3].mean(axis=-1)
        images = {"current frame": frame}
        if self.n_rois:
            images["ROI labels"] = self.labels.astype(np.float32)
        self.summary.set_images(images, selected="current frame")
        self.summary.open()

    # ------------------------------------------------------------------
    # traces and runs, through the movie contract
    # ------------------------------------------------------------------

    def _channel(self) -> int:
        cdim = find_slider_name(self.iw.dim_names, "c")
        return int(self.iw.indices[cdim]) if cdim is not None else 0

    def movie(self, z: int | None = None) -> PlaneMovie | None:
        """``(T, Y, X)`` view of the viewer's array on plane ``z`` (default:
        the plane on screen), or None when the array cannot be wrapped."""
        z = self.z if z is None else int(z)
        try:
            arr = self.iw.data[0]
            nz = PlaneMovie(arr).nz
            return PlaneMovie(arr, z=(z if nz > 1 else 0), c=self._channel())
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    @property
    def trace_busy(self) -> bool:
        self._trace_threads = [t for t in self._trace_threads if t.is_alive()]
        return bool(self._trace_threads)

    def trace_disabled(self, index: int) -> str | None:
        movie = self.movie()
        if movie is None or int(movie.shape[0]) < 2:
            return "no (T, Y, X) movie behind this view"
        return None

    def quick_trace(self, index: int):
        """Mean of the ROI's pixels per frame, on a thread tracked as a job."""
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        if not 0 <= index < self.n_rois:
            return
        record = self.store.rois[index]
        movie = self.movie(record.z)
        if movie is None:
            return
        mask = self.store.labels[record.z] == (index + 1)
        description = f"quick trace - ROI {index}"
        job = get_process_manager().start_job("roi_trace", description)
        self.status = f"{description} started"

        def run():
            try:
                y = roi_trace(movie, mask)
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"{description} failed")
                job.fail(f"{type(error).__name__}: {error}")
                self._trace_results.put((index, None, str(error)))
                return
            job.done(f"{y.size} frames")
            self._trace_results.put((index, {"source": "quick", "F": np.asarray(y, np.float32)}, None))

        thread = threading.Thread(target=run, name=f"roi-trace-{index}", daemon=True)
        self._trace_threads.append(thread)
        thread.start()

    def _show_trace(self, index: int, entry: dict):
        self.traces[index] = entry
        self.trace_roi = index
        self.focus_traces = True

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
        """Drain finished traces and runs; called once per frame from the panel."""
        while True:
            try:
                index, entry, error = self._trace_results.get_nowait()
            except queue.Empty:
                break
            if error is not None:
                self.status = f"trace for ROI {index} failed: {error}"
                continue
            self._show_trace(index, entry)
            self.status = f"ROI {index}: {entry['F'].size} frames"
        outs = self._run_job.poll()
        if outs is not None:
            self.run_status = (
                f"done: {', '.join(Path(o).name for o in outs)}" if outs else "done: nothing written"
            )
            self.logger.info(f"roi run {self.run_status}")
            for out in outs:
                for index, entry in load_run_traces(out).items():
                    self._show_trace(index, entry)
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
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.next_unlabeled()
        if imgui.is_key_pressed(imgui.Key.up_arrow):
            self.step(-1)
        if imgui.is_key_pressed(imgui.Key.down_arrow):
            self.step(1)
        if self.selected >= 0:
            picked = self.classes.hotkey_pressed()
            if picked is not None:
                self.assign_class(picked)

    # ------------------------------------------------------------------
    # imgui: top panel
    # ------------------------------------------------------------------

    def draw_panel(self):
        """Top edge window: tools, overlay, labeling, each gated by its
        Widgets-menu subwidget toggle. Rows wrap and the window grows to fit."""
        self._poll_jobs()
        self.handle_keys()
        with fit_width("ROI tools", min_width=MIN_PANEL_WIDTH) as shown:
            if shown:
                if sub_enabled("manual_roi", "tools"):
                    self._draw_tools_row()
                self._draw_overlay_row()
                if sub_enabled("manual_roi", "labels"):
                    self._draw_labels_row()
        self.summary.draw()
        self.keybinds_open = draw_keybinds_popup(KEYBINDS, self.keybinds_open, "ROI keys")
        if self.tools_window is not None:
            fit_edge_window(self.tools_window, PANEL_HEIGHT)

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
                (None, button_width("Open full FOV"), _fov),
                (None, button_width("keybinds"), _keybinds),
                (None, imgui.calc_text_size("Autosaved").x, self._draw_save_note),
            ]
        )

    def _draw_overlay_row(self):
        box = imgui.get_frame_height()
        dirty = []

        def _masks():
            changed, self.show_masks = imgui.checkbox("masks", self.show_masks)
            dirty.append(changed)

        def _outlines():
            changed, self.show_outlines = imgui.checkbox("outlines", self.show_outlines)
            dirty.append(changed)

        def _opacity():
            changed, self.opacity = imgui.slider_float("##opacity", self.opacity, 0.05, 1.0, "%.2f")
            dirty.append(changed)

        def _status():
            if self._save_error is not None:
                imgui.text_colored(_ERROR_COLOR, self._save_error)
            else:
                imgui.text_disabled(self.status)

        items = [(None, imgui.calc_text_size(self._save_error or self.status).x, _status)]
        if sub_enabled("manual_roi", "overlay"):
            items = [
                (None, button_width("masks") + box, _masks),
                (None, button_width("outlines") + box, _outlines),
                ("roi opacity", 110.0, _opacity),
                *items,
            ]
        draw_toolbar_row(items)
        if any(dirty):
            self.refresh_overlay()

    def _draw_labels_row(self):
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
            f'store = LabelsZarr.load(r"{self._save_target()}")\n'
            "store.labels        # (Z, Y, X) uint16; 0 = bg, ROI i = i + 1\n"
            "store.rois          # per-ROI plane, area, class index, note",
        )
        imgui.end_tooltip()

    # ------------------------------------------------------------------
    # imgui: right tabs
    # ------------------------------------------------------------------

    def draw_tab(self):
        """The ROIs tab: filter row, the ROI table, the selected note, the run row."""
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
            changed, self._note_buf = imgui.input_text_with_hint("##note", "note", self._note_buf)
            if changed:
                self.store.set_note(self.selected, self._note_buf)
            if imgui.is_item_deactivated_after_edit():
                self._autosave()
        else:
            imgui.text_disabled("select an ROI to note")
        if imgui.button("Delete selected"):
            self.delete_roi(self.selected)
        imgui.set_next_item_width(90)
        changed, sel = imgui.combo("##process", PROCESSES.index(self.process), list(PROCESSES))
        if changed:
            self.process = PROCESSES[sel]
        set_tooltip(
            "extract: suite2p-style traces from the drawn masks.\n"
            "demix: masknmf NMF seeded with the drawn masks.\n"
            "Outputs land in rois_<tag>/ beside the data and show up in the Traces tab.",
            show_mark=False,
        )
        imgui.same_line()
        if imgui.button(f"{RUN_ICON} in view"):
            self.run_in_view()
        set_tooltip("Run every ROI currently listed", show_mark=False)
        if self.run_status:
            imgui.same_line()
            imgui.text_disabled(self.run_status)

    @property
    def row_actions(self) -> tuple[RowAction, ...]:
        return (
            RowAction(RUN_ICON, f"Run - {self.process} this ROI", self.run_roi),
            RowAction(TRACE_ICON, "Quick trace - mean of this ROI per frame", self.quick_trace, self.trace_disabled),
        )

    def draw_traces(self):
        """The Traces tab: one ROI's F / Fneu, the cursor bound to the viewer's t."""
        traced = sorted(self.traces)
        if not traced:
            imgui.text_disabled(f"No traces yet. Use {TRACE_ICON} on a row of the ROIs tab, or run extract / demix.")
            return
        if self.trace_roi not in traced:
            self.trace_roi = traced[0]
        imgui.set_next_item_width(100)
        changed, pick = imgui.combo("##trace-roi", traced.index(self.trace_roi), [f"ROI {i}" for i in traced])
        if changed:
            self.trace_roi = traced[pick]
            self.select_roi(self.trace_roi)
        entry = self.traces[self.trace_roi]
        imgui.same_line(0, 10)
        imgui.text_disabled(f"{entry['source']}, frame {self.current_frame()}")
        height = max(imgui.get_content_region_avail().y - 4, 60.0)
        if implot.get_current_context() is None:
            implot.create_context()
        if not implot.begin_plot("##roi_trace_plot", imgui.ImVec2(-1, height), implot.Flags_.no_title):
            return
        try:
            implot.setup_axes("frame", "fluorescence", implot.AxisFlags_.auto_fit, implot.AxisFlags_.auto_fit)
            for name in ("F", "Fneu"):
                if name in entry:
                    implot.plot_line(name, np.ascontiguousarray(entry[name], np.float32))
            if self.tdim is not None:
                moved, frame = implot.drag_line_x(0, float(self.current_frame()), _CURSOR_COLOR, 1.5)[:2]
                if moved:
                    self.set_frame(round(frame))
        finally:
            implot.end_plot()


def attach_roi_widget(parent: Any, focus: bool = False) -> ManualRoiWidget | None:
    """Turn the ROI widget on for a ``PreviewDataWidget``; ROIs from an earlier
    toggle this session are adopted. Returns None (logged) when it cannot be built."""
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
