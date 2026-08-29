"""Manual ROI drawing, labelling and export.

Opened with ``mbo <path> --widget manualroi``, or from Widgets > Manual ROI
Labeling. Every ROI control lives in one edge window across the top — the
same panel masknmf's ``ClassificationVis`` draws, in the same order, off the
same shared widgets — plus a collapsible trace viewer that grows the window
when it is opened. The sortable ROI table is the "ROIs" tab inside
``PreviewDataWidget`` (see ``widgets/tabs.py``).

Arm "Add ROI", drag a closed stroke around a cell, release and the enclosed
pixels become a mask. ROIs live in one uint16 label image so they can never
overlap. Each ROI can be given a class label; 1-9 assign, 0 clears.

The background is the viewer's own image graphic rather than a static FOV,
so the bg source combo picks between the live movie and mean / max / std
projections of the plane on screen. Those same images (plus the movie and
the ROI outlines) open in "Full FOV", the shared ``SummaryImageViewer``.

Each row of the ROI table carries a Quick trace and an Extract trace button.
Both run on their own thread, tracked as a job in the process manager, and
land in the trace viewer, whose cursor is the viewer's own ``t`` — drag it
to scrub the movie.

The drawing, mask, overlay, label, table and summary machinery is shared
with masknmf's classification GUI via ``masknmf.visualization.imgui``.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from imgui_bundle import icons_fontawesome_6 as fa, imgui, implot
from masknmf.visualization.imgui import (
    OUTLINE_PLACEMENT,
    OUTLINE_PLACEMENTS,
    OUTLINE_WIDTH,
    SELECTED_ALPHA,
    UNLABEL_ALL,
    AsyncLoad,
    LabelImage,
    LabelSet,
    LabelStore,
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
)

from mbo_utilities import log
from mbo_utilities.gui._imgui_helpers import selected_button_style, set_tooltip
from mbo_utilities.gui.widgets.widget_toggles import sub_enabled

__all__ = [
    "COLOR_MODES",
    "MAX_OUTLINE_WIDTH",
    "OUTLINE_PLACEMENTS",
    "PANEL_HEIGHT",
    "PANEL_LOCATION",
    "PROJECTIONS",
    "TRACE_HEIGHT",
    "ManualRoiWidget",
    "SELECTED_OPACITY",
]

SELECTED_OPACITY = SELECTED_ALPHA

# the two edge windows, matching ClassificationVis's top panel / side table.
# Four rows: tools, background, masks, labels. The table is narrower than its
# 360 because these ROIs carry three columns rather than six.
PANEL_LOCATION = "top"
PANEL_HEIGHT = 140
# extra height the top window takes while the trace viewer is expanded
TRACE_HEIGHT = 220

COLUMNS = ("id", "label", "area")

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
    ("up / down", "previous / next ROI"),
    ("shift + up / down", "jump 10 ROIs"),
    ("left / right", "previous / next label group"),
    ("1-9", "assign label"),
    ("0", "clear label"),
    ("u", "jump to next unlabeled ROI"),
    ("m", "toggle masks"),
    ("o", "toggle outlines"),
    ("b", "toggle background"),
)

# one saturated color per ROI; hues shuffled so neighbours contrast
_HSV = np.zeros((180, 1, 3), np.uint8)
_HSV[:, 0, 0] = np.random.default_rng(0).permutation(180)
_HSV[:, 0, 1:] = 255
_IDENTITY_COLORS = cv2.cvtColor(_HSV, cv2.COLOR_HSV2RGB).reshape(-1, 3) / 255.0

_CODE_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)
_LOADING_COLOR = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)
_CURSOR_COLOR = imgui.ImVec4(1.0, 0.85, 0.3, 0.9)


class PlaneMovie:
    """A ``(T, Y, X)`` window onto the viewer's array, other dims pinned.

    ``MoviePlayer`` and the projection reducer both want a lazy 3D movie;
    the loaded array is 4D or 5D with z / c sliders in between, so this
    fixes those at whatever the viewer is showing.
    """

    def __init__(self, array, dims, pinned: dict):
        self._array = array
        self._dims = tuple(str(d).upper() for d in dims)
        self.pinned = {str(k).upper(): int(v) for k, v in pinned.items()}
        self._t = self._dims.index("T")
        self._y = self._dims.index("Y")
        self._x = self._dims.index("X")
        shape = array.shape
        self.shape = (shape[self._t], shape[self._y], shape[self._x])

    def _key(self, t, rows, cols) -> tuple:
        key = [self.pinned.get(d, slice(None)) for d in self._dims]
        key[self._t] = t
        key[self._y] = rows
        key[self._x] = cols
        return tuple(key)

    def __getitem__(self, item) -> np.ndarray:
        if isinstance(item, tuple):
            t, rows, cols = item
        else:
            t, rows, cols = item, slice(None), slice(None)
        out = np.asarray(self._array[self._key(t, rows, cols)])
        # a slice on t keeps the frame axis; suite2p's extractor batches
        # with f_in[start:stop], so this has to stay 3D there
        if isinstance(t, slice):
            return out.reshape(-1, *out.shape[-2:])
        return out.reshape(out.shape[-2:])


def mask_trace(movie, mask: np.ndarray) -> np.ndarray:
    """Mean of ``mask``'s pixels per frame, read over the mask's bbox only."""
    rows, cols = np.nonzero(mask)
    if not rows.size:
        return np.zeros(0, np.float32)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    window = mask[y0:y1, x0:x1]
    n_pixels = int(window.sum())
    out = np.empty(int(movie.shape[0]), np.float32)
    for t in range(out.size):
        block = np.asarray(movie[t, slice(y0, y1), slice(x0, x1)], np.float32)
        out[t] = block[window].sum() / n_pixels
    return out


def compute_projections(movie, n_frames: int = PROJECTION_FRAMES) -> dict:
    """Reduce an evenly spaced sample of ``movie`` down to one image each."""
    total = int(movie.shape[0])
    picks = np.unique(np.linspace(0, total - 1, min(total, n_frames)).astype(int))
    stack = np.stack([np.asarray(movie[int(i)], dtype=np.float32) for i in picks])
    return {name: reduce(stack) for name, reduce in PROJECTIONS.items()}


class ManualRoiWidget:
    """Freehand ROI painting and labelling over a ``MboNDViewer``."""

    def __init__(self, iw, fpath=None):
        self.iw = iw
        self.figure = iw.figure
        self.fpath = Path(fpath) if fpath is not None else None
        self.logger = log.get("gui.manual_roi")

        self.subplot = iw.figure[0, 0]
        self.image = iw.graphics[0]
        ny, nx = self.image.data.value.shape[:2]

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
        # px per side of the mask boundary; masknmf's shared default, kept
        # here so it can be tuned per host
        self.outline_width = OUTLINE_WIDTH
        self.outline_alpha = 1.0
        self.placement_idx = OUTLINE_PLACEMENTS.index(OUTLINE_PLACEMENT)
        self.color_mode_idx = 0

        # background: the viewer's own graphic, showing either the live movie
        # or one of the projections
        self.show_bg = True
        self.bg_alpha = 1.0
        self.bg_source_idx = 0
        self._projections: dict[str, np.ndarray] = {}
        self._projection_key: dict | None = None
        self._loader = AsyncLoad()
        self._live_frame: np.ndarray | None = None
        self._frozen_index: dict | None = None
        self._contours: list | None = None

        # per-ROI traces, keyed by ROI index; see quick_trace / extract_trace
        self.traces: dict[int, dict] = {}
        self.trace_open = False
        self.trace_roi = -1
        self._force_traces_open = False
        # worker threads hand finished traces back here, drained on the draw
        # loop; each job is tracked separately in the process manager
        self._trace_results: queue.Queue = queue.Queue()

        self.overlay = self.subplot.add_image(
            np.zeros((ny, nx, 4), np.uint8),
            name="manual_roi_overlay",
            alpha_mode="blend",
            offset=(0, 0, 1),
        )
        # the overlay is literal uint8 RGBA, not data to be contrast-stretched.
        # fastplotlib auto-ranges it off the initial all-zero array, giving
        # vmin == vmax == 0, which saturates every non-zero channel to 255 —
        # tab10 class colours all come out white, and only hues with a zero
        # channel survive. Pin the full byte range instead.
        self.overlay.vmin, self.overlay.vmax = 0, 255
        # keep the overlay out of picking so the tooltip still reports the
        # image intensity under the cursor
        for tile in self.overlay.world_object.children:
            tile.material.pick_write = False

        self.drawer = StrokeDrawer(self.subplot, self.add_roi, self.pick_roi)
        self.summary = SummaryImageViewer(
            self.figure,
            title="Full FOV",
            roi_provider=self.roi_contours,
            on_export=self.export_image,
        )

        # edge windows we own, keyed by location, so teardown never reclaims
        # an edge another widget has taken over since
        self._windows: dict[str, Any] = {}
        self.attach_panel()

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

    # ------------------------------------------------------------------
    # edge windows
    # ------------------------------------------------------------------

    def _add_window(self, draw, location: str, size: int, title: str) -> None:
        add = getattr(self.figure, "add_imgui_window", None)
        if add is None or self._windows.get(location) is not None:
            return
        add(draw, location=location, size=size, title=title)
        self._windows[location] = self.figure.imgui_windows.get(location)

    def _remove_window(self, location: str) -> None:
        window = self._windows.pop(location, None)
        if window is None or self.figure.imgui_windows.get(location) is not window:
            return
        try:
            self.figure.remove_imgui_window(location)
        except Exception:
            self.logger.debug(f"{location} window removal failed", exc_info=True)

    def attach_panel(self) -> None:
        """Hang the control panel off the top edge."""
        self._add_window(self.draw_panel, PANEL_LOCATION, PANEL_HEIGHT, "Manual ROI")

    # ------------------------------------------------------------------
    # ROI state
    # ------------------------------------------------------------------

    def _resync(self):
        self.classes.resize(self.n_rois)
        self.order.columns = {"area": self.masks.areas()}
        self.order.labels = self.classes.labels
        self.order.n_items = self.n_rois
        self.order.set_range_column("area")
        self.order.rebuild()
        self._contours = None

    def add_roi(self, stroke):
        index = self.masks.add(stroke)
        if index < 0:
            self.status = self.masks.last_error or "stroke rejected"
            return
        self._resync()
        self.select_roi(index)

    def pick_roi(self, row: int, col: int):
        self.select_roi(self.masks.at(row, col))

    def select_roi(self, index: int | None):
        self.selected = index if index is not None and 0 <= index < self.n_rois else -1
        self.scroll_to_selection = True
        if self.selected < 0:
            self.status = f"{self.n_rois} ROIs"
        else:
            self.order.goto(self.selected)
            self.status = (
                f"ROI {self.selected + 1}: {self.masks.counts[self.selected]} px"
            )
        self.summary.set_highlight(self.selected_bbox())
        self.refresh_overlay()

    def selected_bbox(self) -> tuple | None:
        """``(y0, x0, h, w)`` of the selected ROI, for the Full FOV highlight."""
        if self.selected < 0:
            return None
        rows, cols = np.nonzero(self.masks.labels == self.selected + 1)
        if not rows.size:
            return None
        y0, x0 = int(rows.min()), int(cols.min())
        return y0, x0, int(rows.max()) - y0 + 1, int(cols.max()) - x0 + 1

    def roi_contours(self) -> list:
        """Outline of every ROI as ``(N, 2)`` y/x points, for the Full FOV."""
        if self._contours is not None:
            return self._contours
        contours = []
        for i in range(self.n_rois):
            mask = (self.masks.labels == i + 1).astype(np.uint8)
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
        if not self.masks.delete(index):
            return
        # deleting renumbers every ROI above it, so cached traces no longer
        # belong to the indices that key them
        self.traces.clear()
        self._resync()
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index + 1}"

    def clear(self):
        self.masks.clear()
        self.traces.clear()
        self._resync()
        self.select_roi(-1)
        self.status = "cleared"

    def label_selected(self, label_index: int):
        if self.selected < 0:
            return
        self.classes.assign([self.selected], label_index)
        self._after_labelling()

    def unlabel_all(self):
        self.classes.clear()
        self._after_labelling()
        self.status = f"cleared {self.n_rois} labels"

    def _after_labelling(self):
        self.save_labels()
        self.order.rebuild()
        self.refresh_overlay()

    @property
    def color_mode(self) -> str:
        return COLOR_MODES[self.color_mode_idx]

    @property
    def outline_placement(self) -> str:
        return OUTLINE_PLACEMENTS[self.placement_idx]

    def _colors(self) -> np.ndarray:
        """One RGB per ROI, per the colour mode.

        ``label`` gives every ROI in a class the same colour and greys the
        unlabeled; ``roi`` gives each its own hue so neighbours stay apart
        while drawing; ``label + roi`` does the first where a label exists
        and the second where it does not.
        """
        mode = self.color_mode
        colors = np.zeros((max(self.n_rois, 1), 3), np.float32)
        for i in range(self.n_rois):
            label = int(self.classes.labels[i])
            identity = _IDENTITY_COLORS[i % len(_IDENTITY_COLORS)]
            if mode == "roi":
                colors[i] = identity
            elif label >= 0:
                colors[i] = self.classes.color(label)
            else:
                colors[i] = identity if mode == "label + roi" else UNLABELED_COLOR
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
            edges=self.masks.edges(self.outline_width, self.outline_placement),
            outline_width=self.outline_width,
            outline_alpha=self.outline_alpha,
            outline_placement=self.outline_placement,
        )

    # ------------------------------------------------------------------
    # background and summary images
    # ------------------------------------------------------------------

    @property
    def bg_sources(self) -> list[str]:
        """The live movie, then the projections. Reduced on first selection."""
        return ["movie", *PROJECTIONS]

    def plane_movie(self) -> PlaneMovie | None:
        """The loaded array as ``(T, Y, X)`` at the viewer's current z / c."""
        try:
            array = self.iw.data[0]
        except (AttributeError, IndexError, TypeError):
            return None
        dims = getattr(array, "dims", None)
        if dims is None:
            # a plain ndarray, as MboNDViewer(data=...) accepts: the NDWidget
            # convention is slider dims first, then the image rows and columns
            dims = (*self.iw.slider_dims, "Y", "X")
        dims = tuple(str(d).upper() for d in dims)
        if not {"T", "Y", "X"}.issubset(dims):
            return None
        if len(dims) != getattr(array, "ndim", len(dims)):
            return None
        pinned = {
            str(dim).upper(): int(index)
            for dim, index in dict(self.iw.current_index).items()
            if str(dim).upper() in dims and str(dim).upper() not in ("T", "Y", "X")
        }
        return PlaneMovie(array, dims, pinned)

    def drop_stale_projections(self):
        """Forget projections reduced from a plane the viewer has left."""
        movie = self.plane_movie()
        if movie is None or self._projection_key is None:
            return
        if movie.pinned != self._projection_key:
            self._projections = {}
            self._projection_key = None

    def request_projections(self):
        """Kick off the projection reduce for the plane on screen, once."""
        self.drop_stale_projections()
        movie = self.plane_movie()
        if movie is None or self._loader.busy or self._projections:
            return
        self._projection_key = dict(movie.pinned)
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
        projection is gone the moment the user scrubs; the combo has to say so.
        The projections themselves stay cached — they only depend on the
        pinned z / c, which ``request_projections`` checks for itself.
        """
        if self._frozen_index is None:
            return
        if dict(self.iw.current_index) != self._frozen_index:
            self._frozen_index = None
            self.bg_source_idx = 0

    def open_full_fov(self):
        """Open the shared summary viewer over the projections and the movie."""
        self.request_projections()
        movie = self.plane_movie()
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

    # ------------------------------------------------------------------
    # traces
    # ------------------------------------------------------------------

    def trace_extractor(self):
        """The pipeline "Extract trace" would run, or None."""
        from mbo_utilities.gui.widgets.pipelines import get_trace_extractors

        extractors = get_trace_extractors()
        return extractors[0] if extractors else None

    def trace_disabled(self, index: int) -> str | None:
        """Why the trace actions cannot run for ``index``, or None."""
        if self.plane_movie() is None:
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

    def _start_trace(self, index: int, source: str, work):
        """Run ``work`` on a thread, tracked as a process-manager job.

        Every click gets its own job and its own thread: the button is the
        only feedback the user has otherwise, and two ROIs can be extracting
        at once. Results come back through a queue and are picked up by
        :meth:`_poll_trace` on the draw loop.
        """
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        description = f"{source} trace — ROI {index + 1}"
        job = get_process_manager().start_job("roi_trace", description)
        self.status = f"{description} started"

        def run():
            try:
                result = work()
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"{description} failed")
                job.fail(f"{type(error).__name__}: {error}")
                self._trace_results.put((index, source, None, str(error)))
                return
            if not result:
                job.fail("the extractor returned nothing")
                self._trace_results.put((index, source, None, "no trace returned"))
                return
            frames = int(np.asarray(next(iter(result.values()))).shape[-1])
            job.done(f"{frames} frames")
            self._trace_results.put((index, source, result, None))

        threading.Thread(
            target=run, name=f"roi-trace-{index}", daemon=True
        ).start()

    def quick_trace(self, index: int):
        """Mean of the ROI's pixels per frame — no pipeline, just the mask."""
        movie = self.plane_movie()
        if movie is None or not 0 <= index < self.n_rois:
            return
        mask = self.masks.labels == index + 1
        self._start_trace(
            index,
            "quick",
            lambda: {"F": mask_trace(movie, mask)[None, :]},
        )

    def extract_trace(self, index: int):
        """Hand the mask to a pipeline's own extractor (suite2p, say)."""
        movie = self.plane_movie()
        pipeline = self.trace_extractor()
        if movie is None or pipeline is None or not 0 <= index < self.n_rois:
            return
        # one ROI at a time: the pipeline sees only this mask, so its
        # neuropil ring is not carved up by the others
        labels = (self.masks.labels == index + 1).astype(np.uint16)
        self._start_trace(
            index,
            pipeline.name,
            lambda: pipeline.extract_traces(movie, labels),
        )

    def _poll_trace(self):
        """Drain finished trace jobs; called once per frame from the panel."""
        while True:
            try:
                index, source, result, error = self._trace_results.get_nowait()
            except queue.Empty:
                return
            if error is not None:
                self.status = f"{source} trace failed: {error}"
                continue
            self.traces[index] = {
                "source": source,
                **{k: np.asarray(v, np.float32) for k, v in result.items()},
            }
            self.trace_roi = index
            self.status = f"{source} trace for ROI {index + 1}"
            # a finished trace is the point of the click, so show it
            self.trace_open = True
            self._force_traces_open = True

    # ------------------------------------------------------------------
    # the trace viewer, a collapsible section of the top panel
    # ------------------------------------------------------------------

    def current_frame(self) -> int:
        return int(dict(self.iw.current_index).get("t", 0))

    def set_frame(self, frame: int):
        """Move the viewer's t, so dragging the cursor scrubs the movie."""
        movie = self.plane_movie()
        limit = (int(movie.shape[0]) - 1) if movie is not None else 0
        self.iw.current_index = {"t": int(np.clip(frame, 0, limit))}

    def traced_rois(self) -> list[int]:
        return sorted(self.traces)

    def _sync_panel_height(self):
        """Grow the top edge window to fit the trace viewer, shrink when closed."""
        window = self._windows.get(PANEL_LOCATION)
        if window is None:
            return
        wanted = PANEL_HEIGHT + (TRACE_HEIGHT if self.trace_open else 0)
        if window.size != wanted:
            window.size = wanted

    def draw_trace_section(self):
        """Collapsible trace viewer, its cursor locked to the viewer's t."""
        traced = self.traced_rois()
        label = f"Traces ({len(traced)})###roi_traces" if traced else "Traces###roi_traces"
        if self._force_traces_open:
            imgui.set_next_item_open(True)
            self._force_traces_open = False
        expanded = imgui.collapsing_header(label)
        if expanded != self.trace_open:
            self.trace_open = expanded
            self._sync_panel_height()
        if not expanded:
            return

        if not traced:
            imgui.text_disabled(
                f"No traces yet. Use the {QUICK_TRACE_ICON} or "
                f"{EXTRACT_TRACE_ICON} button on a row of the ROIs tab."
            )
            return

        if self.trace_roi not in traced:
            self.trace_roi = traced[0]
        names = [f"ROI {i + 1}" for i in traced]
        imgui.set_next_item_width(90)
        changed, pick = imgui.combo(
            "##trace-roi", traced.index(self.trace_roi), names
        )
        if changed:
            self.trace_roi = traced[pick]
        entry = self.traces[self.trace_roi]
        imgui.same_line(0, 10)
        imgui.text_disabled(f"source: {entry['source']}")
        imgui.same_line(0, 10)
        if imgui.small_button("save npz"):
            self.save_trace(self.trace_roi)
        imgui.same_line(0, 10)
        imgui.text_disabled(f"frame {self.current_frame()}")

        height = max(imgui.get_content_region_avail().y - 4, 60.0)
        if not implot.begin_plot("##roi_trace_plot", imgui.ImVec2(-1, height)):
            return
        implot.setup_axes(
            "frame",
            "fluorescence (a.u.)",
            implot.AxisFlags_.auto_fit.value,
            implot.AxisFlags_.auto_fit.value,
        )
        for name in ("F", "Fneu"):
            values = entry.get(name)
            if values is None:
                continue
            trace = np.asarray(values, np.float64).reshape(-1)
            implot.plot_line(name, np.arange(trace.size, dtype=np.float64), trace)
        # the viewer's own t, as a guide you can drag to scrub
        moved, frame = implot.drag_line_x(
            0, float(self.current_frame()), _CURSOR_COLOR, 1.5
        )[:2]
        if moved:
            self.set_frame(round(frame))
        implot.end_plot()

    def save_trace(self, index: int):
        entry = self.traces.get(index)
        if entry is None:
            return
        out = self._output_path(f"roi{index + 1}_trace.npz")
        arrays = {k: v for k, v in entry.items() if k != "source"}
        try:
            np.savez(out, source=np.array(entry["source"]), **arrays)
        except OSError as error:
            self.status = f"trace save failed: {error}"
            return
        self.status = f"saved {out.name}"

    def _row_actions(self) -> tuple[RowAction, ...]:
        return (
            RowAction(
                QUICK_TRACE_ICON,
                "Quick trace — average this ROI's pixels over time",
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

    def _extract_tooltip(self) -> str:
        pipeline = self.trace_extractor()
        which = pipeline.name if pipeline is not None else "a pipeline"
        return f"Extract trace — run this ROI's mask through {which}"

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
    # persistence
    # ------------------------------------------------------------------

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
        hotkey = self.classes.hotkey_pressed()
        if hotkey is not None:
            self.label_selected(hotkey)

    def set_drawing(self, on: bool):
        self.drawer.arm(on)
        self.status = (
            "drag a closed stroke around a cell" if on else f"{self.n_rois} ROIs"
        )

    def close(self):
        """Drop both edge windows, the pointer handlers and the graphics.

        Everything here has to go for the widget to be rebuildable: a
        ``remove_graphic`` leaves the name taken, and a live ``StrokeDrawer``
        keeps handling pointer events against a dead line graphic.
        """
        self.set_drawing(False)
        self.set_bg_source(0)
        self.show_bg = True
        self.bg_alpha = 1.0
        self.apply_background()
        self.trace_open = False
        self.summary.close()
        self.summary.cleanup()
        self._remove_window(PANEL_LOCATION)
        renderer = self.subplot.renderer
        for handler, kind in (
            (self.drawer._down, "pointer_down"),
            (self.drawer._move, "pointer_move"),
            (self.drawer._up, "pointer_up"),
        ):
            try:
                renderer.remove_event_handler(handler, kind)
            except Exception:
                self.logger.debug(f"{kind} handler removal failed", exc_info=True)
        for graphic in (self.overlay, self.drawer.line):
            try:
                self.subplot.delete_graphic(graphic)
            except Exception:
                self.logger.debug("overlay removal failed", exc_info=True)

    # ------------------------------------------------------------------
    # panels
    # ------------------------------------------------------------------

    def _draw_save_note(self):
        """Where masks and labels land, in ClassificationVis's save-note slot."""
        imgui.text_disabled("Autosaved")
        if not imgui.is_item_hovered():
            return
        imgui.begin_tooltip()
        imgui.text(f"labels: {self.store.npz_path}")
        imgui.text(f"masks:  {self._output_path('manual_masks.npy')} (on Save)")
        imgui.text_colored(
            _CODE_COLOR,
            "import numpy as np\n"
            "\n"
            'masks  = np.load(r"manual_masks.npy")  # (Y, X) uint16; 0 = bg, ROI i = i + 1\n'
            'data   = np.load(r"manual_labels.npz")\n'
            'names  = data["label_names"]           # class names; row index = label value\n'
            'labels = data["class_labels"]          # (num_rois,) int64; -1 = unlabeled',
        )
        imgui.end_tooltip()

    def _draw_tools_row(self):
        with selected_button_style(self.drawer.armed):
            if imgui.button("Add ROI"):
                self.set_drawing(not self.drawer.armed)
        set_tooltip(
            "Drag a closed stroke around a cell, release to fill it. "
            "'a' toggles, esc stops, ctrl+Z undoes.",
            show_mark=False,
        )
        imgui.same_line(0, 5)
        if imgui.button("Undo"):
            self.delete_roi(self.n_rois - 1)
        imgui.same_line(0, 5)
        if imgui.button("Clear"):
            self.clear()
        imgui.same_line(0, 5)
        if imgui.button("Save"):
            self.save()
        imgui.same_line(0, 20)
        if imgui.button("prev"):
            self.step(-1)
        imgui.same_line(0, 5)
        if imgui.button("next"):
            self.step(1)
        imgui.same_line(0, 10)
        imgui.set_next_item_width(160)
        changed, pos = imgui.slider_int(
            "##pos", self.order.pos, 0, max(len(self.order.order) - 1, 0)
        )
        if changed and len(self.order.order):
            self.order.pos = pos
            self.select_roi(self.order.current)
        imgui.same_line(0, 20)
        if imgui.button("Open full FOV"):
            self.open_full_fov()
        imgui.same_line(0, 20)
        self._draw_save_note()
        imgui.same_line(max(imgui.get_window_width() - 90, 0))
        if imgui.button("keybinds"):
            self.keybinds_open = True

    def _draw_background_row(self):
        changed_bg, self.show_bg = imgui.checkbox("##bg-on", self.show_bg)
        imgui.same_line(0, 6)
        imgui.set_next_item_width(110)
        changed_src, index = imgui.combo(
            "##bg-source", self.bg_source_idx, self.bg_sources
        )
        if changed_src:
            self.set_bg_source(index)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The viewer's own image. Projections are reduced from an "
                "evenly spaced sample of the plane on screen; moving a "
                "slider hands the image back to the movie."
            )
        imgui.same_line(0, 6)
        imgui.text("bg image")
        imgui.same_line(0, 4)
        imgui.text_disabled("(b)")
        imgui.same_line(0, 12)
        imgui.set_next_item_width(75)
        changed_bga, self.bg_alpha = imgui.slider_float(
            "bg opacity", self.bg_alpha, 0.0, 1.0, "%.2f"
        )
        if changed_bg or changed_bga:
            self.apply_background()

    def _draw_mask_row(self):
        show_masks, self.show_masks = imgui.checkbox("masks", self.show_masks)
        imgui.same_line(0, 4)
        imgui.text_disabled("(m)")
        imgui.same_line(0, 12)
        imgui.set_next_item_width(70)
        fill, self.opacity = imgui.slider_float("fill", self.opacity, 0.05, 1.0, "%.2f")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Opacity of the mask interiors.")

        imgui.same_line(0, 20)
        show_outlines, self.show_outlines = imgui.checkbox(
            "outlines", self.show_outlines
        )
        imgui.same_line(0, 4)
        imgui.text_disabled("(o)")
        imgui.same_line(0, 12)
        imgui.set_next_item_width(70)
        width, self.outline_width = imgui.slider_int(
            "px", self.outline_width, 1, MAX_OUTLINE_WIDTH
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Outline thickness. The overlay is one texel per data pixel, "
                "so 1 px is the thinnest a line can be drawn."
            )
        imgui.same_line(0, 8)
        imgui.set_next_item_width(80)
        placement, self.placement_idx = imgui.combo(
            "##placement", self.placement_idx, list(OUTLINE_PLACEMENTS)
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Which side of the boundary the outline sits on.\n"
                "outer:  on the background just outside the mask — costs no\n"
                "        mask pixel, so 1-3 px structures keep every pixel\n"
                "inner:  on the mask's own outermost pixels\n"
                "center: straddles the boundary, eating a pixel each way"
            )
        imgui.same_line(0, 12)
        imgui.set_next_item_width(70)
        line, self.outline_alpha = imgui.slider_float(
            "line", self.outline_alpha, 0.05, 1.0, "%.2f"
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("Opacity of the outlines, independent of the fill.")

        imgui.same_line(0, 20)
        imgui.set_next_item_width(100)
        color, self.color_mode_idx = imgui.combo(
            "color by", self.color_mode_idx, list(COLOR_MODES)
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "label:       one colour per class, unlabeled grey\n"
                "label + roi: class colour where labelled, a unique hue otherwise\n"
                "roi:         a unique hue per ROI"
            )
        if show_masks or fill or show_outlines or width or placement or line or color:
            self.refresh_overlay()

    def _draw_status(self):
        if self._loader.busy:
            imgui.text_colored(_LOADING_COLOR, self._loader.status)
        else:
            imgui.text_disabled(self._loader.error or self.status)

    def _draw_labels_row(self):
        if self.n_rois:
            if draw_progress(self.classes.labels):
                self.next_unlabeled()
            imgui.same_line(0, 24)
        self.new_label, changed = draw_label_editor(
            self.classes, self.new_label, "_roi"
        )
        picked = draw_label_buttons(self.classes, "_roi")
        if picked == UNLABEL_ALL:
            self.unlabel_all()
        elif picked is not None:
            self.label_selected(picked)
        if changed:
            self.order.rebuild()
            self.refresh_overlay()

    def draw_panel(self):
        """Top edge window: every ROI control, in ClassificationVis's order."""
        self._poll_projections()
        self._poll_trace()
        self.drop_stale_projections()
        self._follow_viewer()
        self.handle_keys()

        if sub_enabled("manual_roi", "tools"):
            self._draw_tools_row()
        if sub_enabled("manual_roi", "overlay"):
            self._draw_background_row()
            imgui.same_line(0, 20)
            self._draw_status()
            self._draw_mask_row()
        else:
            self._draw_status()
        if sub_enabled("manual_roi", "labels"):
            self._draw_labels_row()

        self.draw_trace_section()
        self.summary.draw()
        self.keybinds_open = draw_keybinds_popup(
            KEYBINDS, self.keybinds_open, "ROI keys"
        )

    def draw_table(self):
        """The ROIs tab: the filter row over the sortable ROI table.

        Hosted by ``PreviewDataWidget``'s tab bar (see ``widgets/tabs.py``),
        beside Preview and Signal Quality, rather than by an edge window.
        """
        draw_filter_row(self.order, self.classes, "_roi")
        if not self.n_rois:
            imgui.text_disabled("no ROIs yet")
            return
        self.scroll_to_selection = draw_roi_table(
            self.order,
            self.classes,
            COLUMNS,
            {"area": lambda i: f"{self.masks.counts[i]}"},
            self.scroll_to_selection,
            table_id="manual_rois",
            on_select=self.select_roi,
            actions=self._row_actions(),
        )
