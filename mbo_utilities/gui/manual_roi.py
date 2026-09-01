"""Manual ROI drawing + labeling widget for the viewer.

Toggled from ``Widgets > Manual ROI Labeling`` in the preview GUI (``mbo
<path> --widget manualroi`` opens with it on). A strip of control cards
across the top of the figure - NAVIGATE, DRAW, VIEW, LABELS, PROCESS, each
gated by its Widgets-menu subwidget toggle - sits over a status row, and
three tabs in the right-hand widget (``widgets/tabs.py``) hold the combined
ROI table, the trace viewer, and the run browser. The table, label set,
stroke capture, overlay compositing and theme are the shared widgets from
``mbo_utilities.gui.imgui``.

Arm "Add ROI" (a), drag a closed stroke around a cell, release and the
enclosed pixels become a mask. ROIs live in a ``RoiLabelStore``: one
``(P, Y, X)`` uint16 label volume (0 is background, ROI ``i`` is ``i + 1``)
so they can never overlap. Each ROI keeps a persistent ``uid`` and a
``source`` naming where it came from ("" = drawn by hand). A stroke lands
on the exact slice the viewer shows: every scrolling dim except time (z,
channel, any extra slider) keys its own plane of masks, and flipping a
slider swaps the overlays, table filter and traces to that slice's ROIs.
Data without scroll sliders degrades to a single plane. Annotations
autosave next to the data as an OME-NGFF-style labels zarr
(``manual_labels.zarr``, see ``mbo_utilities.annotation``) and are restored
from it on relaunch.

Runs happen in place: the PROCESS card sends the listed ROIs through
extract / demix (``rois_<tag>/`` beside the data), detects new ROIs inside
a dragged region (r), or spawns a full suite2p / masknmf plane. Loaded run
outputs become derived sets - a second overlay plus rows in the table -
whose components can be promoted into the drawn store (y) or discarded (n).
Loaded runs are remembered in a ``roi_runs.json`` sidecar and restored on
relaunch. Traces (quick per-ROI means and run outputs) are keyed by store
uid, so deleting an ROI never remaps anyone else's rows.

With drawing off, clicking selects what is under the cursor - a derived
component when its overlay shows there, else the drawn ROI - and clicking
the background clears the selection. Ctrl+click (in the image, the ROI
table, or the trace-plot legend) toggles the ROI in a group buffer and
shift+click adds to it (a row range in the table); class labels and the
group color then apply to every member, so two cells can be grouped and
sent to "soma" in two clicks. Mask, table and trace colors all come from
the same per-ROI color. Selecting a listed ROI on another plane jumps
every slider that plane encodes. Only the first subplot is drawable.
"""

from __future__ import annotations

import queue
import textwrap
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from imgui_bundle import (
    imgui,
    imgui_ctx,
    icons_fontawesome_6 as fa,
    implot,
)
from mbo_utilities.gui.imgui import (
    UNLABEL_ALL,
    LabelSet,
    RoiOrder,
    RowAction,
    StrokeDrawer,
    SummaryImageViewer,
    draw_label_editor,
    draw_label_filter,
    draw_range_filter,
    draw_roi_table,
)
from mbo_utilities import log
from mbo_utilities.gui._top_strip import TopPanel, TopStrip
from mbo_utilities.annotation import UNLABELED, LabelsZarr, RoiLabelStore
from mbo_utilities.arrays.features import find_slider_name
from mbo_utilities.gui._imgui_helpers import (
    fit_width,
    selected_button_style,
    set_tooltip,
)
from mbo_utilities.gui._keyboard import claim_arrow_keys
from mbo_utilities.gui._theme import (
    THEME,
    card,
    close_button,
    danger_button,
    em,
    label_button,
    popup,
    section,
    to_vec4,
)
from mbo_utilities.gui.roi_runs import (
    SELECTED_ALPHA,
    DerivedSet,
    RoiRun,
    RoiRunManager,
    TraceSet,
    component_color,
    derived_rgba,
    display_fneu,
    display_trace,
    feathered_rgba,
    finished_dirs,
    full_plane_args,
    load_run_registry,
    registry_path,
    run_dir_complete,
    save_run_registry,
    set_color,
)
from mbo_utilities.gui.widgets.process_manager import get_process_manager
from mbo_utilities.gui.widgets.widget_toggles import sub_enabled
from mbo_utilities.roi_workflow import (
    OUT_PREFIX,
    PlaneMovie,
    demix_rois,
    discover_rois,
    extract_rois,
    feather_mask,
    load_run_dir,
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
# height the ROI / Traces bodies ask the top strip for (the strip adds the
# menu row and its tab bar on top of this)
PANEL_HEIGHT = 200

# a card narrower than this clips its controls, so instead of squeezing them
# the row wraps and the panel asks the strip for another row's height
MIN_CARD_EM = 20.0
MAX_CARD_ROWS = 3
# below this width the tabs collapse to a placeholder line
MIN_TAB_WIDTH = 150

# Traces tab columns: (name, stretch weight, hidden by default). The order is
# the sort-key order in draw_trace_table, so only ever append.
TRACE_COLUMNS = (
    ("roi", 1.6, False),
    ("source", 1.8, False),
    ("frames", 1.0, True),
    ("mean", 1.0, False),
    ("peak", 1.0, True),
    ("snr", 1.0, False),
)

MIN_ROI_PIXELS = 9
MIN_REGION_SIDE = 4
SELECTED_OPACITY = SELECTED_ALPHA
SAVE_NAME = "manual_labels.zarr"
# no seeded class labels: the label set starts empty, the user names their own
DEFAULT_LABEL_NAMES: tuple[str, ...] = ()
COLUMNS = ("id", "label", "source", "ok")
PROCESSES = ("extract", "extract-s2p", "demix")

RUN_ICON = fa.ICON_FA_PLAY
TRACE_ICON = fa.ICON_FA_CHART_LINE
REMOVE_ICON = fa.ICON_FA_XMARK

KEYBINDS = (
    ("a", "arm / disarm ROI drawing"),
    ("r", "arm / disarm region drawing"),
    ("esc", "stop drawing, else clear the region"),
    ("ctrl+z", "undo the last drawn ROI"),
    ("delete", "delete the selected ROI / discard the selected algo one"),
    ("up / down", "previous / next trace on the Traces panel, else ROI in view"),
    ("u", "next unlabeled ROI"),
    ("f", "center the shown ROI; labeling then advances"),
    ("1-9", "label the selected ROI (drawn or algo), then advance"),
    ("0", "clear its label"),
    ("y", "promote the selected algo ROI"),
    ("n", "discard the selected algo ROI"),
    ("x", "accept / reject the selected algo ROI"),
    ("t", "quick trace the selected ROI"),
    ("b", "toggle the drawn overlay"),
    ("d", "toggle the algo overlay"),
    ("click", "select what is under the cursor (drawing off)"),
    ("ctrl+click", "toggle an ROI in the group (image, table, trace legend)"),
    ("shift+click", "add to the group (a row range in the table)"),
    ("esc", "also empties the group"),
)

_HELP_STEPS = (
    "Arm Add ROI (a) and drag a closed stroke around a cell; release fills "
    "it. Ctrl+Z undoes, delete removes the selection.",
    "Label ROIs with the class buttons or keys 1-9 (0 clears); u jumps to "
    "the next unlabeled one and labeling steps there on its own.",
    "PROCESS runs the listed ROIs through extract (suite2p-style traces), "
    "extract-s2p (suite2p's extractor) or demix (masknmf seeded NMF).",
    "Draw a region with r, then find masknmf / find suite2p detects ROIs "
    "inside it, unseeded.",
    "Detected components arrive as an algo overlay and table rows: "
    "promote one into the drawn set (y) or discard it (n). Deleting a "
    "promoted ROI makes its row promotable again.",
    "The Traces tab plots quick traces and run traces per ROI; the Runs "
    "tab lists active, finished, loaded and on-disk runs.",
)
_HELP_FILES = (
    "manual_labels.zarr  the drawn ROIs, autosaved\n"
    "                    (mbo_utilities.annotation.LabelsZarr.load)\n"
    "rois_<tag>/         one run's outputs: stat.npy, F.npy, Fneu.npy,\n"
    "                    iscell.npy, ops.npy, rois.json\n"
    "roi_runs.json       which runs this dataset has loaded"
)

_CURSOR_COLOR = imgui.ImVec4(1.0, 0.85, 0.3, 0.9)
_FNEU_COLOR = (0.25, 0.55, 1.0, 1.0)
_fneu_cmap: int | None = None


def _fneu_colormap() -> int:
    """The registered single-blue colormap every neuropil line draws with."""
    global _fneu_cmap
    if _fneu_cmap is None:
        idx = implot.get_colormap_index("mbo_fneu")
        if idx < 0:
            idx = implot.add_colormap(
                "mbo_fneu", np.array([_FNEU_COLOR, _FNEU_COLOR], np.float32)
            )
        _fneu_cmap = int(idx)
    return _fneu_cmap


def _line_colormap(rgb) -> int:
    """A registered single-color colormap for one trace line (this implot
    build has no per-line color argument, so lines take their color from the
    pushed colormap). Looked up by name so a recreated context re-registers."""
    key = tuple(int(round(float(v) * 255)) for v in rgb)
    name = "mbo_line_{}_{}_{}".format(*key)
    idx = implot.get_colormap_index(name)
    if idx < 0:
        color = (key[0] / 255.0, key[1] / 255.0, key[2] / 255.0, 1.0)
        idx = implot.add_colormap(name, np.array([color, color], np.float32))
    return int(idx)


def card_grid(n: int, avail: float, min_w: float, gap: float) -> tuple[int, int, float]:
    """Lay ``n`` cards out across ``avail`` px.

    Parameters
    ----------
    n : int
        Number of cards to place.
    avail : float
        Width available to the row, in pixels.
    min_w : float
        Narrowest a card may be drawn; cards wrap onto another row rather
        than shrink past it, so their contents never clip.
    gap : float
        Horizontal space between two cards.

    Returns
    -------
    tuple of (int, int, float)
        Cards per row, number of rows, and the width every card is drawn at.
        Rows are evened out, so five cards three-to-a-row go 3 + 2 rather
        than leaving a single card on the second row.
    """
    per_row = min(max(int((avail + gap) // (min_w + gap)), 1), max(n, 1))
    rows = -(-n // per_row)
    per_row = -(-n // rows)
    return per_row, rows, max((avail - gap * (per_row - 1)) / per_row, min_w)


def labels_path(fpath) -> Path:
    """``manual_labels.zarr`` beside a file, or inside a directory."""
    base = Path.cwd() if fpath is None else Path(fpath)
    return (base.parent if base.suffix else base) / SAVE_NAME


def roi_widgets_available() -> bool:
    """True when the shared imgui widgets import."""
    try:
        import mbo_utilities.gui.imgui  # noqa: F401
    except Exception:
        return False
    return True


def _cleared_note(cleared) -> str:
    """Status tail naming the filters a selection had to drop to show itself."""
    return f" · cleared the {', '.join(cleared)} filter" + ("s" if len(cleared) > 1 else "") if cleared else ""


class _PlaneOrder(RoiOrder):
    """``RoiOrder`` with "only this z-plane" and "only this source" filters."""

    def __init__(self, columns, labels, n_items):
        super().__init__(columns, labels, n_items)
        self.plane: int | None = None
        self.planes = np.zeros(0, np.int64)
        self.source: int | None = None  # None = all, 0 = drawn, 1 + si = a set
        self.sources = np.zeros(0, np.int64)

    def rebuild(self):
        super().rebuild()
        if not len(self.order):
            return
        keep = np.ones(len(self.order), bool)
        if self.plane is not None:
            keep &= self.planes[self.order] == self.plane
        if self.source is not None:
            keep &= self.sources[self.order] == self.source
        if keep.all():
            return
        current = self.current
        self.order = self.order[keep]
        hits = np.flatnonzero(self.order == current)
        self.pos = int(hits[0]) if len(hits) else int(
            min(self.pos, max(len(self.order) - 1, 0))
        )

    def hidden_by(self, item: int) -> list:
        out = super().hidden_by(item)
        if self.plane is not None and int(self.planes[item]) != self.plane:
            out.append("plane")
        if self.source is not None and int(self.sources[item]) != self.source:
            out.append("source")
        return out

    def clear_filter(self, name: str):
        if name == "plane":
            self.plane = None
        elif name == "source":
            self.source = None
        else:
            super().clear_filter(name)

    def next_unlabeled(self) -> bool:
        # only drawn rows can take a label, so u never lands on a derived one
        hits = np.flatnonzero(
            (self.labels[self.order] < 0) & (self.sources[self.order] == 0)
        )
        if not len(hits):
            return False
        after = hits[hits > self.pos]
        self.pos = int(after[0] if len(after) else hits[0])
        return True


class ManualRoiWidget:
    """Freehand ROI painting, labeling and run curation on a ``MboNDViewer``.

    Parameters
    ----------
    iw : MboNDViewer
        The viewer to draw on. Only its first subplot is drawable.
    fpath : path-like, optional
        The data path; annotations autosave to ``manual_labels.zarr`` beside
        it, loaded runs are remembered in ``roi_runs.json``, and both are
        restored from there on construction.
    label_names : iterable of str
        Class labels to seed the label set with.
    store : RoiLabelStore, optional
        Adopt this in-memory store instead of starting empty / restoring
        from disk (how ROIs survive an off/on toggle).
    runs : dict, optional
        State from :meth:`park_runs` of the previous widget (how loaded
        runs, traces and live background work survive an off/on toggle).
    strip : TopStrip, optional
        The figure's shared top strip to hang the ROI and Traces panels off.
        Omit to own one (standalone use).
    host : PreviewDataWidget, optional
        The widget that owns the viewer's display settings; the trace plot
        follows its window function. Omit when running standalone.
    """

    def __init__(self, iw, fpath=None, label_names=(), store=None, runs=None, strip=None, host=None):
        self.iw = iw
        self.host = host
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
        self.cdim = find_slider_name(iw.dim_names, "c")
        # every scrolling dim except time keys its own mask plane, so masks
        # follow the channel / z / any extra slider; z sits last in the flat
        # order so a z-only store keeps plane == z and old stores restore
        axes = []
        for name in iw.dim_names:
            if name == self.tdim:
                continue
            rr = iw.ndwidget.indices.ref_ranges.get(name)
            n = max(int(rr.stop - rr.start), 1) if rr is not None else 1
            if n > 1:
                axes.append((name, n))
        axes.sort(key=lambda a: a[0] == self.zdim)
        self.plane_axes: tuple[tuple[str, int], ...] = tuple(axes)
        nz = int(np.prod([n for _, n in axes])) if axes else 1

        self._adopted_store = store is not None and (store.nz, store.ny, store.nx) == (nz, self.ny, self.nx)
        if self._adopted_store:
            self.store = store
        else:
            self.store = RoiLabelStore(nz, self.ny, self.nx, min_pixels=MIN_ROI_PIXELS)
        self.store.plane_axes = self.plane_axes
        for name in label_names:
            self.store.add_label_name(name)

        self.selected = -1
        self.selected_derived: tuple[int, int] | None = None
        # ctrl / shift click builds a group here; label and color actions
        # then apply to every member. Entries are (si, k), si -1 = drawn.
        self.buffer: list[tuple[int, int]] = []
        self._group_color = (1.0, 0.8, 0.2)
        self._pending_row_action: tuple[str, int, int] | None = None
        self.status = "press Add ROI to start"
        self._save_error: str | None = None
        self._run_error: str | None = None
        self._writer: LabelsZarr | None = None
        self.new_label = ""
        self._note_buf = ""
        self.help_open = False
        self.keybinds_open = False
        self.scroll_to_selection = False
        self.follow = False  # center the shown ROI; labeling then advances

        self.show_masks = True
        self.opacity = 0.45
        self.show_derived = True
        self.derived_opacity = 0.6

        self._feathers: dict[int, tuple] = {}
        self.rows: list[tuple[int, int]] = []
        self._row_index: dict[tuple[int, int], int] = {}
        self._promoted: dict[tuple[str, int], int] = {}
        self.derived: list[DerivedSet] = []
        self.classes = LabelSet(0, self.store.label_names)
        self.order = _PlaneOrder({"source": np.zeros(0, np.int64)}, self.classes.labels, 0)

        self.z = self._current_z()
        if self.plane_axes:
            iw.ndwidget.indices.add_event_handler(self._on_indices)

        self.overlay = self.subplot.add_image(
            np.zeros((self.ny, self.nx, 4), np.uint8),
            name="manual_roi_overlay",
            alpha_mode="blend",
            offset=(0, 0, 1),
        )
        self.derived_overlay = self.subplot.add_image(
            np.zeros((self.ny, self.nx, 4), np.uint8),
            name="manual_roi_derived",
            alpha_mode="blend",
            offset=(0, 0, 1.5),
        )
        # literal RGBA bytes: auto-ranging off the all-zero start saturates every colour to white
        for overlay in (self.overlay, self.derived_overlay):
            overlay.vmin, overlay.vmax = 0, 255
            for tile in overlay.world_object.children:
                tile.material.pick_write = False
        self.derived_overlay.visible = False

        self.drawer = StrokeDrawer(self.subplot, self._on_stroke, self._pick)
        self.summary = SummaryImageViewer(iw.figure, title="Full FOV")
        self.region: tuple[int, int, int, int] | None = None
        self.region_mode = False
        self.region_line = None

        # traces keyed by store uid per origin; see quick_trace and _poll_jobs
        self.trace_sets: dict[str, TraceSet] = {}
        self.trace_uid = 0
        self._trace_results: queue.Queue = queue.Queue()
        self._trace_threads: list[threading.Thread] = []
        self.trace_sel: set[tuple] = set()  # trace-table keys to plot
        self._trace_stats: dict[tuple, tuple] = {}
        self._trace_display: dict[tuple, tuple] = {}
        # one entry: the last windowed line, so panning does not recompute it
        self._trace_window_cache: dict[tuple, np.ndarray] = {}
        self.correct_neuropil = True
        self._trace_sort = (0, True)
        self._trace_fit = True
        self._plot_key = None

        self.process = PROCESSES[0]
        self.run_tag = "manual"
        self.manager = RoiRunManager()
        self._registry_extra: list[dict] = []
        self._restoring = False

        # the top edge is shared (menu row, Signal Quality plot, these two
        # panels); standalone use — tests, a bare viewer — gets its own strip
        self._own_strip = strip is None
        self.tools_window = TopStrip(iw.figure) if self._own_strip else strip
        self.tools_window.add_hook(self._frame)
        # kept so the panel can ask for more height once its cards wrap
        self._roi_panel = TopPanel(
            "roi", "ROI", self._draw_roi_panel, PANEL_HEIGHT, "rois", 10
        )
        self.tools_window.register(self._roi_panel)
        self.tools_window.register(
            TopPanel("traces", "Traces", self.draw_traces, PANEL_HEIGHT, "traces", 11)
        )

        self._closed = False
        self._restore()
        self._restore_runs(runs)
        self._resync()
        self.refresh_derived_overlay()

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
        if self.plane_axes:
            try:
                self.iw.ndwidget.indices.remove_event_handler(self._on_indices)
            except (KeyError, ValueError, AttributeError):
                pass
        graphics = [self.overlay, self.derived_overlay, self.drawer.line]
        if self.region_line is not None:
            graphics.append(self.region_line)
        for graphic in graphics:
            try:
                self.subplot.delete_graphic(graphic)
            except (KeyError, ValueError):
                pass
        try:
            self.summary.close()
            self.summary.cleanup()
        except Exception:
            self.logger.debug("summary viewer cleanup failed", exc_info=True)
        self.tools_window.remove_hook(self._frame)
        self.tools_window.unregister("roi")
        self.tools_window.unregister("traces")
        if self._own_strip:
            self.tools_window.close()
        self.tools_window = None

    def park_runs(self) -> dict:
        """State handed to the next attach: the live run manager, loaded
        sets, registry leftovers and the uid-keyed traces."""
        return {
            "manager": self.manager,
            "derived": self.derived,
            "extra": self._registry_extra,
            "trace_sets": self.trace_sets,
        }

    # ------------------------------------------------------------------
    # z / t tracking
    # ------------------------------------------------------------------

    def _current_z(self) -> int:
        """Flat store plane for the viewer's scroll position.

        Plain z for z-only data; with channels or other scroll dims each
        combination owns a plane (``plane_axes``, z fastest-varying), so the
        masks on screen always belong to the exact slice on screen.
        """
        if not self.plane_axes:
            return 0
        sizes = [n for _, n in self.plane_axes]
        idx = [
            int(np.clip(self.iw.indices[name], 0, n - 1))
            for name, n in self.plane_axes
        ]
        return int(np.ravel_multi_index(idx, sizes))

    def _plane_pos(self, plane: int) -> dict[str, int]:
        """``{dim name: index}`` behind one flat store plane."""
        if not self.plane_axes:
            return {}
        sizes = [n for _, n in self.plane_axes]
        vals = np.unravel_index(int(np.clip(plane, 0, self.store.nz - 1)), sizes)
        return {name: int(v) for (name, _n), v in zip(self.plane_axes, vals)}

    def _goto_plane(self, plane: int):
        """Move every scroll slider so the viewer shows ``plane``."""
        for name, v in self._plane_pos(plane).items():
            if int(self.iw.indices[name]) != v:
                self.iw.indices[name] = v

    def _plane_label(self, plane: int) -> str:
        """``"2"`` for plain z planes, ``"c1·z2"`` when more dims key them."""
        pos = self._plane_pos(plane)
        if not pos:
            return "1"
        if len(pos) == 1:
            return f"{next(iter(pos.values())) + 1}"
        return "·".join(f"{name}{v + 1}" for name, v in pos.items())

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
        self.refresh_derived_overlay()

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
        return self.drawer.armed and not self.region_mode

    @property
    def stroke(self) -> list:
        return self.drawer.stroke

    @property
    def stroke_line(self):
        return self.drawer.line

    def _resync(self):
        """Rebuild the combined rows, label set and table order from the
        store and the loaded derived sets. Drawn rows come first so table
        ids match store indices; promoted rows are recomputed from the
        store's ``source`` strings."""
        rois = self.store.rois
        self._promoted = {}
        for i, r in enumerate(rois):
            name, _, row = r.source.rpartition(":")
            if name and row.isdigit():
                self._promoted[(name, int(row))] = i
        self.rows = [(-1, i) for i in range(len(rois))]
        planes = [r.z for r in rois]
        sources = [0] * len(rois)
        oks = [1] * len(rois)
        for si, s in enumerate(self.derived):
            if not s.visible:
                continue
            for k, stat_row in enumerate(s.result.stat):
                if k in s.discarded:
                    continue
                self.rows.append((si, k))
                planes.append(s.result.z)
                sources.append(1 + si)
                oks.append(1 if s.accepted[k] else 0)
        self._row_index = {pair: row for row, pair in enumerate(self.rows)}
        if self.selected_derived is not None and self.selected_derived not in self._row_index:
            self.selected_derived = None
        self.buffer = [pair for pair in self.buffer if pair in self._row_index]
        labels = np.full(len(self.rows), UNLABELED, np.int64)
        labels[: len(rois)] = [r.class_index for r in rois]
        for row in range(len(rois), len(self.rows)):
            si, k = self.rows[row]
            labels[row] = self.derived[si].classes.get(k, UNLABELED)
        self.classes = LabelSet(len(self.rows), self.store.label_names, labels)
        self.store.label_names = self.classes.names
        planes = np.asarray(planes, np.int64)
        columns = {
            "source": np.asarray(sources, np.int64),
            "ok": np.asarray(oks, np.int64),
        }
        if self.store.nz > 1:
            columns["z"] = planes
        self.order.columns = columns
        self.order.labels = self.classes.labels
        self.order.n_items = len(self.rows)
        self.order.planes = planes
        self.order.sources = columns["source"]
        if self.order.source is not None and not 0 <= self.order.source <= len(self.derived):
            self.order.source = None
        self.order.rebuild()

    def _sync_store_from_classes(self):
        self.store.label_names = tuple(self.classes.names)
        for record, ci in zip(self.store.rois, self.classes.labels):
            record.class_index = int(ci)
        for row in range(self.n_rois, len(self.rows)):
            si, k = self.rows[row]
            ci = int(self.classes.labels[row])
            if ci == UNLABELED:
                self.derived[si].classes.pop(k, None)
            else:
                self.derived[si].classes[k] = ci

    @property
    def columns(self) -> tuple[str, ...]:
        return COLUMNS + (("z",) if self.store.nz > 1 else ())

    def _formatters(self) -> dict:
        def source(row):
            si, k = self.rows[row]
            if si < 0:
                return "drawn"
            s = self.derived[si]
            return f"{s.name} · promoted" if (s.name, k) in self._promoted else s.name

        def zplane(row):
            si, k = self.rows[row]
            z = self.store.rois[k].z if si < 0 else self.derived[si].result.z
            return self._plane_label(z)

        def ok(row):
            si, k = self.rows[row]
            if si < 0:
                return ""
            return "yes" if self.derived[si].accepted[k] else "no"

        return {"source": source, "ok": ok, "z": zplane}

    # ------------------------------------------------------------------
    # mask state
    # ------------------------------------------------------------------

    def _arm_mode(self, mode: str):
        """One of "off", "roi", "region" owns the stroke drawer."""
        current = "region" if self.region_mode else ("roi" if self.drawer.armed else "off")
        if mode == current:
            return
        self.region_mode = mode == "region"
        self.drawer.arm(mode != "off")
        if mode == "roi":
            self.status = "drag a closed stroke around a cell"
        elif mode == "region":
            self.status = "drag a box around the region"
        else:
            self.status = f"{self.n_rois} ROIs"

    def set_drawing(self, on: bool):
        """Arm or disarm ROI stroke drawing (lifts the pan binding while armed)."""
        self._arm_mode("roi" if on else "off")

    def set_region_mode(self, on: bool):
        """Arm or disarm region drawing; a finished drag becomes ``self.region``."""
        self._arm_mode("region" if on else "off")

    def _on_stroke(self, stroke):
        # runs inside a renderer pointer event: a raise here would vanish
        # into the event loop and could leave a stored ROI undrawn
        try:
            if self.region_mode:
                self._set_region(stroke)
            else:
                self.add_roi(stroke)
        except Exception as e:  # noqa: BLE001 - surfaced in the status row
            self.logger.exception("stroke handling failed")
            self.status = f"stroke failed: {type(e).__name__}: {e}"
            self._resync()
            self.refresh_overlay()

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

    def _set_region(self, stroke):
        if len(stroke) < 2:
            return
        points = np.asarray(stroke, np.float32)
        y0 = int(np.clip(np.floor(points[:, 1].min()), 0, self.ny - 1))
        x0 = int(np.clip(np.floor(points[:, 0].min()), 0, self.nx - 1))
        y1 = int(np.clip(np.ceil(points[:, 1].max()), y0 + 1, self.ny))
        x1 = int(np.clip(np.ceil(points[:, 0].max()), x0 + 1, self.nx))
        if y1 - y0 < MIN_REGION_SIDE or x1 - x0 < MIN_REGION_SIDE:
            self.status = f"region under {MIN_REGION_SIDE} px per side, ignored"
            return
        self.region = (y0, y1, x0, x1)
        if self.region_line is None:
            self.region_line = self.subplot.add_line(
                np.zeros((5, 3), np.float32), colors="cyan", thickness=1.5,
                name="roi_region", offset=(0, 0, 1.75), visible=False,
            )
            self.region_line.world_object.material.pick_write = False
        self.region_line.data = np.array(
            [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0], [x0, y0, 0]],
            np.float32,
        )
        self.region_line.visible = True
        self.status = f"region {y1 - y0}x{x1 - x0}"

    def clear_region(self):
        self.region = None
        if self.region_line is not None:
            self.region_line.visible = False

    def _pick(self, row: int, col: int, mods: frozenset = frozenset()):
        """Select what the click shows: a visible derived component first
        (the derived overlay draws on top), else the drawn ROI. Ctrl+click
        toggles it in the group buffer, shift+click adds it; a plain click
        replaces the group with the single selection."""
        try:
            hit: tuple[int, int] | None = None
            if self.show_derived and 0 <= row < self.ny and 0 <= col < self.nx:
                for si, s in enumerate(self.derived):
                    if not s.visible or s.result.z != self.z:
                        continue
                    k = int(s.pick_map[row, col])
                    if k >= 0 and k not in s.discarded:
                        hit = (si, k)
                        break
            if hit is None:
                index = self.store.roi_at(self.z, row, col)
                if index >= 0:
                    hit = (-1, index)
            if "Ctrl" in mods:
                if hit is not None:
                    self.buffer_toggle(*hit)
                return
            if "Shift" in mods:
                if hit is not None:
                    self.buffer_add(*hit)
                return
            self.buffer_clear()
            if hit is None:
                self.select_roi(-1)
            elif hit[0] < 0:
                self.select_roi(hit[1])
            else:
                self.select_derived(*hit)
        except Exception as e:  # noqa: BLE001 - surfaced in the status row
            self.logger.exception("pick failed")
            self.status = f"pick failed: {type(e).__name__}: {e}"

    def _row_grouped(self, item: int) -> bool:
        return 0 <= item < len(self.rows) and self.rows[item] in self.buffer

    def _table_select(self, item: int):
        """Plain table click: single selection, group dropped."""
        self.buffer_clear()
        self.select_row(item)

    def _table_ctrl(self, item: int):
        if 0 <= item < len(self.rows):
            self.buffer_toggle(*self.rows[item])

    def select_row(self, row: int | None):
        """Route a table-row selection to the right kind."""
        if row is None or not 0 <= row < len(self.rows):
            self.select_roi(-1)
            return
        si, k = self.rows[row]
        if si < 0:
            self.select_roi(k)
        else:
            self.select_derived(si, k)

    def select_roi(self, index: int | None):
        """Select drawn ROI ``index``; anything out of range clears the
        selection (and any derived one).

        Selecting an ROI on another plane jumps the z slider to it; one with
        a trace shows it in the Traces tab.
        """
        self.selected_derived = None
        self.selected = index if index is not None and 0 <= index < self.n_rois else -1
        self.scroll_to_selection = True
        if self.selected < 0:
            self._note_buf = ""
            self.status = f"{self.n_rois} ROIs"
        else:
            record = self.store.rois[self.selected]
            self._note_buf = record.note
            cleared = self.order.reveal(self.selected)
            self.status = f"ROI {self.selected}: {record.area} px" + _cleared_note(cleared)
            if record.z != self.z:
                self._goto_plane(record.z)
            if any(record.uid in ts.data for ts in self.trace_sets.values()):
                self.trace_uid = record.uid
            if self.follow:
                self._center_on(*self._feather(self.selected)[:2])
        self._sync_trace_sel()
        self.refresh_overlay()
        self.refresh_derived_overlay()

    def select_derived(self, si: int, k: int):
        """Select component ``k`` of derived set ``si`` (clears any drawn
        selection; jumps z to the set's plane)."""
        if not (0 <= si < len(self.derived) and 0 <= k < len(self.derived[si].result.stat)):
            self.select_roi(-1)
            return
        self.selected = -1
        self._note_buf = ""
        self.selected_derived = (si, k)
        self.scroll_to_selection = True
        s = self.derived[si]
        row = self._row_index.get((si, k))
        cleared = self.order.reveal(row) if row is not None else []
        stat_row = s.result.stat[k]
        npix = int(stat_row.get("npix", len(stat_row["ypix"])))
        tail = " · promoted" if (s.name, k) in self._promoted else ""
        self.status = f"{s.name} row {k}: {npix} px{tail}" + _cleared_note(cleared)
        if s.result.z != self.z:
            self._goto_plane(s.result.z)
        if self.follow:
            self._center_on(stat_row["ypix"], stat_row["xpix"])
        self._sync_trace_sel()
        self.refresh_overlay()
        self.refresh_derived_overlay()

    # ------------------------------------------------------------------
    # group buffer (ctrl / shift multi-select)
    # ------------------------------------------------------------------

    def in_buffer(self, si: int, k: int) -> bool:
        return (si, k) in self.buffer

    def _seed_buffer(self):
        """A first ctrl / shift click keeps the current selection grouped,
        so 'select one, ctrl+click another' makes a group of two."""
        if self.buffer:
            return
        if self.selected >= 0:
            self.buffer.append((-1, self.selected))
        elif self.selected_derived is not None:
            self.buffer.append(self.selected_derived)

    def buffer_add(self, si: int, k: int):
        """Add one row to the group and make it the shown one."""
        self._seed_buffer()
        if (si, k) not in self.buffer:
            self.buffer.append((si, k))
        self._after_buffer_change(si, k)

    def buffer_toggle(self, si: int, k: int):
        """Ctrl+click: flip one row's group membership."""
        self._seed_buffer()
        if (si, k) in self.buffer:
            self.buffer.remove((si, k))
            self._refresh_group_view()
            self.status = f"{len(self.buffer)} in group"
        else:
            self.buffer.append((si, k))
            self._after_buffer_change(si, k)

    def buffer_extend_to(self, item: int):
        """Shift+click in the table: group every row between the cursor and
        ``item``, in the order the table shows."""
        hits = np.flatnonzero(self.order.order == item)
        if not len(hits):
            return
        self._seed_buffer()
        a, b = sorted((self.order.pos, int(hits[0])))
        for pos in range(a, b + 1):
            pair = self.rows[int(self.order.order[pos])]
            if pair not in self.buffer:
                self.buffer.append(pair)
        self._after_buffer_change(*self.rows[item])

    def buffer_clear(self):
        if self.buffer:
            self.buffer = []
            self._refresh_group_view()

    def _after_buffer_change(self, si: int, k: int):
        n = len(self.buffer)
        if si < 0:
            self.select_roi(k)
        else:
            self.select_derived(si, k)
        if n > 1:
            self.status = f"{n} in group · labels and color apply to all"

    def _refresh_group_view(self):
        self.refresh_overlay()
        self.refresh_derived_overlay()

    def set_group_color(self, rgb: tuple[float, float, float] | None):
        """Give every grouped ROI (or just the selection) an explicit
        display color, in masks, table and traces alike; None reverts to
        the class / hue colors."""
        targets = list(self.buffer)
        if not targets:
            if self.selected >= 0:
                targets = [(-1, self.selected)]
            elif self.selected_derived is not None:
                targets = [self.selected_derived]
        if not targets:
            return
        rgb255 = None if rgb is None else tuple(int(round(float(v) * 255)) for v in rgb)
        for si, k in targets:
            if si < 0:
                self.store.set_color(k, rgb255)
            elif rgb is None:
                self.derived[si].colors.pop(k, None)
            else:
                self.derived[si].colors[k] = tuple(float(v) for v in rgb)
        self._refresh_group_view()
        self._autosave()
        self._save_registry()
        self.status = (
            f"colored {len(targets)} ROI(s)" if rgb is not None
            else f"reset {len(targets)} color(s)"
        )

    def step(self, delta: int):
        """Up / down: the next or previous trace while the Traces panel is
        up, else the next or previous ROI. Either way the image, both tables
        and the trace plot land on the same ROI."""
        if self.top_tab == "traces" and self.step_trace(delta):
            return
        if self.order.step(delta):
            self.select_row(self.order.current)

    def step_trace(self, delta: int) -> bool:
        """Move to the next / previous row of the trace table, in the order
        the table shows, and select the ROI behind it."""
        rows = self._sorted_trace_rows()
        if not rows:
            return False
        at = next((i for i, key in enumerate(rows) if key in self.trace_sel), None)
        if at is None:
            pos = 0 if delta > 0 else len(rows) - 1
        else:
            pos = int(np.clip(at + delta, 0, len(rows) - 1))
        self.select_trace(rows[pos])
        return True

    def select_trace(self, key):
        """Plot just this trace and select the ROI behind it."""
        self.trace_sel = {key}
        self._trace_fit = True
        origin, name, k = key
        if origin == "uid":
            index = self.store.uid_index(k)
            if index is not None:
                self.trace_uid = k
                self.select_roi(index)
        else:
            hit = self._set_by_name(name)
            if hit is not None:
                self.select_derived(hit[0], k)

    def toggle_trace(self, key):
        """Add / remove one trace from the plotted set (ctrl+click)."""
        (self.trace_sel.discard if key in self.trace_sel else self.trace_sel.add)(key)
        self._trace_fit = True

    def next_unlabeled(self):
        if self.order.next_unlabeled():
            self.select_row(self.order.current)

    def _select_next_derived(self, start_pos: int, skip_promoted: bool = False):
        """Select the next derived row in view at or after ``start_pos``."""
        for pos in range(max(start_pos, 0), len(self.order.order)):
            row = int(self.order.order[pos])
            si, k = self.rows[row]
            if si < 0:
                continue
            if skip_promoted and (self.derived[si].name, k) in self._promoted:
                continue
            self.select_row(row)
            return

    def delete_roi(self, index: int):
        """Drop one drawn ROI and renumber the labels above it; traces of
        every other ROI survive (they are keyed by uid)."""
        if not self.store.delete_roi(index):
            return
        live = {r.uid for r in self.store.rois}
        for ts in self.trace_sets.values():
            ts.prune(live)
        self._feathers = {u: v for u, v in self._feathers.items() if u in live}
        # drawn indices above the deleted one shift down by one
        self.buffer = [
            (si, k - (1 if si < 0 and k > index else 0))
            for si, k in self.buffer
            if not (si < 0 and k == index)
        ]
        self._traces_changed()
        self._resync()
        self.select_roi(min(index, self.n_rois - 1))
        self.status = f"deleted ROI {index}"
        self._autosave()

    def delete_selected(self):
        """Delete the selected drawn ROI, or discard the selected derived one."""
        if self.selected >= 0:
            self.delete_roi(self.selected)
        elif self.selected_derived is not None:
            self.discard_derived(*self.selected_derived, advance=True)

    def clear(self):
        self.store.clear()
        self.trace_sets.clear()
        self._feathers.clear()
        self.buffer = []
        self._traces_changed()
        self._resync()
        self.select_roi(-1)
        self.status = "cleared"
        self._autosave()

    def assign_class(self, class_index: int):
        """Give the selected ROI - drawn or derived - a class label;
        UNLABELED (-1) clears it. With a group of two or more (ctrl / shift
        click) the label lands on every member."""
        if len(self.buffer) > 1:
            for si, k in self.buffer:
                if si < 0:
                    self.store.set_class(k, class_index)
                elif class_index == UNLABELED:
                    self.derived[si].classes.pop(k, None)
                else:
                    self.derived[si].classes[k] = int(class_index)
            self._resync()
            name = (
                "unlabeled" if class_index == UNLABELED
                else self.classes.names[class_index]
            )
            self.status = f"{len(self.buffer)} ROIs -> {name}"
            self._refresh_group_view()
            self._autosave()
            self._save_registry()
            return
        if self.selected >= 0:
            self.store.set_class(self.selected, class_index)
            self._resync()
            self.order.goto(self.selected)
            self.status = f"ROI {self.selected}: {self.classes.name_of(self.selected)}"
            self.refresh_overlay()
            self._autosave()
            if self.follow and class_index != UNLABELED:
                self.next_unlabeled()
            return
        if self.selected_derived is None:
            return
        si, k = self.selected_derived
        s = self.derived[si]
        if class_index == UNLABELED:
            s.classes.pop(k, None)
        else:
            s.classes[k] = int(class_index)
        self._resync()
        row = self._row_index.get((si, k))
        if row is not None:
            self.order.goto(row)
            self.status = f"{s.name} row {k}: {self.classes.name_of(row)}"
        self._save_registry()
        if self.follow and class_index != UNLABELED:
            self._select_next_derived(self.order.pos + 1)

    label_selected = assign_class

    def unlabel_all(self):
        for i in range(self.n_rois):
            self.store.set_class(i, UNLABELED)
        self.classes.assign(range(self.n_rois), UNLABELED)
        self.order.rebuild()
        self.refresh_overlay()
        self.status = f"cleared {self.n_rois} labels"
        self._autosave()

    def _feather(self, index: int) -> tuple:
        """``(ypix, xpix, lam)`` of one drawn mask, soft-edged; cached by
        uid (a mask's pixels never change once drawn)."""
        record = self.store.rois[index]
        got = self._feathers.get(record.uid)
        if got is None:
            mask = self.store.labels[record.z] == index + 1
            w = feather_mask(mask)
            ypix, xpix = np.nonzero(mask)
            got = (ypix.astype(np.int32), xpix.astype(np.int32), w[ypix, xpix])
            self._feathers[record.uid] = got
        return got

    def refresh_overlay(self):
        """Drawn masks, feathered and colored exactly like the imported
        ones; only the table's source column tells them apart."""
        self.overlay.visible = self.show_masks
        if not self.overlay.visible:
            return
        comps = []
        sel = None
        grouped = {k for si, k in self.buffer if si < 0}
        for i, record in enumerate(self.store.rois):
            if record.z != self.z:
                continue
            ypix, xpix, lam = self._feather(i)
            rgb = np.asarray(self.store.roi_rgb(i), np.float32) / 255.0
            fill = SELECTED_OPACITY if i in grouped else self.opacity
            comps.append((ypix, xpix, lam, rgb, fill))
            if i == self.selected:
                sel = (ypix, xpix, rgb)
        self.overlay.data = feathered_rgba((self.ny, self.nx), comps, sel)

    def _center_on(self, ypix, xpix):
        """Frame the camera on one mask with some context around it."""
        if not len(ypix):
            return
        y0, y1 = float(np.min(ypix)), float(np.max(ypix))
        x0, x1 = float(np.min(xpix)), float(np.max(xpix))
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        half = max(y1 - y0, x1 - x0, 1.0) * 2.0
        half = max(half, 40.0)
        self.subplot.camera.show_rect(cx - half, cx + half, cy - half, cy + half)

    def _center_selection(self):
        if self.selected >= 0:
            ypix, xpix, _lam = self._feather(self.selected)
            self._center_on(ypix, xpix)
        elif self.selected_derived is not None:
            si, k = self.selected_derived
            row = self.derived[si].result.stat[k]
            self._center_on(row["ypix"], row["xpix"])

    def toggle_follow(self):
        """Flip review mode: the shown ROI is framed by the camera and a
        label steps to the next one, like masknmf's classification GUI."""
        self.follow = not self.follow
        if self.follow:
            self._center_selection()
            self.status = "centering on the shown ROI; labeling advances"
        else:
            self.status = f"{self.n_rois} ROIs"

    def toggle_drawn_overlay(self):
        self.show_masks = not self.show_masks
        self.refresh_overlay()

    # ------------------------------------------------------------------
    # derived sets
    # ------------------------------------------------------------------

    def refresh_derived_overlay(self):
        """Recompute the derived overlay for the plane on screen; called on
        select / z / load / discard / promote / toggle / opacity changes."""
        sets_on_z = [s for s in self.derived if s.result.z == self.z]
        show = self.show_derived and bool(sets_on_z)
        self.derived_overlay.visible = show
        if not show:
            return
        selected = None
        if self.selected_derived is not None:
            si, k = self.selected_derived
            if self.derived[si].result.z == self.z:
                selected = (self.derived[si], k)
        grouped = {(id(self.derived[si]), k) for si, k in self.buffer if si >= 0}
        self.derived_overlay.data = derived_rgba(
            (self.ny, self.nx), sets_on_z, self.derived_opacity, selected,
            grouped=grouped,
        )

    def toggle_derived_overlay(self):
        self.show_derived = not self.show_derived
        self.refresh_derived_overlay()

    def _set_name(self, path: Path) -> str:
        """Display name for a run dir; per-plane ``zNN`` children keep the
        run dir they belong to."""
        path = Path(path)
        name = path.name
        if name[:1] == "z" and name[1:].isdigit():
            name = f"{path.parent.name}/{name}"
        while any(s.name == name for s in self.derived):
            name += "~"
        return name

    def _add_derived(self, res, discarded=(), classes=None, colors=None) -> DerivedSet | None:
        """Wrap a loaded run as a derived set (replacing an earlier load of
        the same dir) and merge its uid-keyed traces."""
        if tuple(res.shape) != (self.ny, self.nx):
            self._run_error = (
                f"{res.path.name} is {res.shape[0]}x{res.shape[1]}, "
                f"data is {self.ny}x{self.nx}"
            )
            return None
        if not 0 <= res.z < self.store.nz:
            self._run_error = (
                f"{res.path.name} is plane {res.z + 1}, "
                f"data has {self.store.nz} plane(s)"
            )
            return None
        promoted_traces: dict = {}
        classes = dict(classes or {})
        colors = dict(colors or {})
        for si, old in enumerate(self.derived):
            if old.result.path == res.path:
                discarded = set(discarded) | old.discarded
                classes = {**old.classes, **classes}
                colors = {**old.colors, **colors}
                ts = self.trace_sets.get(old.name)
                if ts is not None:
                    uids = {
                        r.uid for r in self.store.rois
                        if r.source.rsplit(":", 1)[0] == old.name
                    }
                    promoted_traces = {u: e for u, e in ts.data.items() if u in uids}
                self.unload_set(si)
                break
        s = DerivedSet(res, self._set_name(res.path), set_color(len(self.derived)),
                       discarded={int(k) for k in discarded},
                       classes={int(k): int(v) for k, v in classes.items()},
                       colors={int(k): tuple(float(x) for x in v)
                               for k, v in colors.items()})
        self.derived.append(s)
        self._merge_run_traces(res, s.name)
        if promoted_traces:
            ts = self.trace_sets.setdefault(s.name, TraceSet(s.name, res.kind))
            ts.data.update(promoted_traces)
        self._traces_changed()
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()
        return s

    def load_run(self, path, discarded=(), classes=None, colors=None) -> bool:
        """Read one run dir into the widget: extract runs merge their
        traces, everything else loads as a derived set (every row, the
        rejected ones included - curation happens here)."""
        path = Path(path)
        try:
            res = load_run_dir(path, iscell_only=False, logger=self.logger)
        except Exception as e:  # noqa: BLE001 - shown in the status row
            self._run_error = f"could not load {path.name}: {e}"
            return False
        if (
            self.store.nz == 1 and res.z != 0
            and self.fpath is not None and path == labels_path(self.fpath).parent
        ):
            # the movie on screen IS this plane, whatever z the run recorded
            res = replace(res, z=0)
        if res.kind == "extract":
            if self._merge_run_traces(res, self._set_name(path)):
                self.focus_traces = True
            self._traces_changed()
            if not any(str(e["path"]) == str(path) for e in self._registry_extra):
                self._registry_extra.append(
                    {"path": str(path), "kind": res.kind, "discarded": []}
                )
            self._save_registry()
            return True
        return self._add_derived(res, discarded, classes, colors) is not None

    def unload_set(self, si: int):
        s = self.derived.pop(si)
        self.trace_sets.pop(s.name, None)
        self._traces_changed()
        self._registry_extra = [
            e for e in self._registry_extra if str(e["path"]) != str(s.result.path)
        ]
        if self.selected_derived is not None:
            osi, k = self.selected_derived
            if osi == si:
                self.selected_derived = None
            elif osi > si:
                self.selected_derived = (osi - 1, k)
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()

    def promoted_index(self, si: int, k: int) -> int | None:
        """Store index of the drawn ROI promoted from set ``si`` row ``k``."""
        return self._promoted.get((self.derived[si].name, k))

    def _promote(self, si: int, k: int) -> int | None:
        """Copy one derived component into the store; None with a status
        message when it cannot land."""
        s = self.derived[si]
        if k in s.discarded:
            self.status = f"{s.name} row {k} is discarded"
            return None
        if (s.name, k) in self._promoted:
            self.status = f"{s.name} row {k} is already promoted"
            return None
        if len(self.store.rois) >= 65535:
            self.status = "store is full (65535 labels)"
            return None
        stat_row = s.result.stat[k]
        mask = np.zeros((self.ny, self.nx), bool)
        mask[stat_row["ypix"], stat_row["xpix"]] = True
        index = self.store.add_roi(s.result.z, mask, source=f"{s.name}:{k}")
        if index is None:
            self.status = "overlaps existing ROIs, nothing free to claim"
            return None
        self._promoted[(s.name, k)] = index
        if k in s.classes:
            self.store.set_class(index, s.classes[k])
        entry = self._derived_entry(s.result, k)
        if entry is not None:
            ts = self.trace_sets.setdefault(s.name, TraceSet(s.name, s.result.kind))
            ts.data[self.store.rois[index].uid] = entry
            self._traces_changed()
        return index

    def promote_derived(self, si: int, k: int) -> int | None:
        """Promote one derived component, select it, then step to the next
        promotable derived row in view."""
        index = self._promote(si, k)
        if index is None:
            return None
        self._resync()
        self.select_roi(index)
        self.refresh_overlay()
        self.refresh_derived_overlay()
        self._autosave()
        row = self._row_index.get((si, k))
        start = 0
        if row is not None:
            hits = np.flatnonzero(self.order.order == row)
            if len(hits):
                start = int(hits[0]) + 1
        self._select_next_derived(start, skip_promoted=True)
        return index

    def promote_set(self, si: int):
        """Promote every shown component of one set."""
        s = self.derived[si]
        promoted = skipped = 0
        for k in range(len(s.result.stat)):
            if k in s.discarded or (s.name, k) in self._promoted:
                skipped += 1
                continue
            if self._promote(si, k) is None:
                skipped += 1
            else:
                promoted += 1
        self._resync()
        self.refresh_overlay()
        self.refresh_derived_overlay()
        self._autosave()
        self.status = f"{s.name}: promoted {promoted} / skipped {skipped}"

    def set_accepted(self, si: int, k: int, on: bool | None = None):
        """Flip (or set) one derived component's accepted flag, mirrored
        into the run dir's ``iscell.npy``."""
        s = self.derived[si]
        s.accepted[k] = (not s.accepted[k]) if on is None else bool(on)
        path = s.result.path / "iscell.npy"
        try:
            n = len(s.result.stat)
            iscell = np.load(path) if path.exists() else np.ones((n, 2), np.float32)
            if len(iscell) != n:
                iscell = np.ones((n, 2), np.float32)
            iscell[k, 0] = 1.0 if s.accepted[k] else 0.0
            np.save(path, iscell)
        except OSError as e:
            self._save_error = f"iscell save failed: {e}"
        self._resync()
        self.refresh_derived_overlay()
        state = "accepted" if s.accepted[k] else "rejected"
        self.status = f"{s.name} row {k}: {state}"

    def discard_derived(self, si: int, k: int, advance: bool = False):
        """Hide one derived component (undone from the Runs tab)."""
        s = self.derived[si]
        s.discarded.add(int(k))
        if self.selected_derived == (si, k):
            self.selected_derived = None
        self._traces_changed()
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()
        self.status = f"discarded {s.name} row {k}"
        if advance:
            self._select_next_derived(self.order.pos)

    def undiscard_derived(self, si: int, k: int):
        self.derived[si].discarded.discard(int(k))
        self._traces_changed()
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()

    def restore_discarded(self, si: int):
        self.derived[si].discarded.clear()
        self._traces_changed()
        self._resync()
        self.refresh_derived_overlay()
        self._save_registry()

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
        if self._adopted_store:
            # the parked store is the in-session truth; the zarr can be
            # behind it when an autosave failed
            return
        if not target.exists():
            return
        try:
            store = LabelsZarr.load(target)
        except (OSError, ValueError) as e:
            self.logger.warning(f"could not restore {target}: {e}")
            self.status = f"restore failed: {e}"
            return
        if (store.nz, store.ny, store.nx) != (self.store.nz, self.store.ny, self.store.nx):
            zsize = dict(self.plane_axes).get(self.zdim, 1)
            if (
                (store.ny, store.nx) == (self.store.ny, self.store.nx)
                and not store.plane_axes
                and store.nz == zsize < self.store.nz
            ):
                # a store saved before channels keyed planes: z is last in
                # the flat order, so its planes are the first ones here
                grown = np.zeros(self.store.labels.shape, np.uint16)
                grown[: store.nz] = store.labels
                store = RoiLabelStore(
                    self.store.nz, self.store.ny, self.store.nx,
                    label_names=store.label_names, labels=grown,
                    rois=store.rois, next_uid=store.next_uid,
                )
            else:
                self.logger.warning(
                    f"{target} is {store.labels.shape}, data wants "
                    f"{self.store.labels.shape}; starting fresh"
                )
                self.status = "saved labels do not match this data, starting fresh"
                return
        for name in self.store.label_names:
            store.add_label_name(name)
        store.min_pixels = MIN_ROI_PIXELS
        store.plane_axes = self.plane_axes
        self.store = store
        self.status = f"restored {len(store.rois)} ROIs"
        self.refresh_overlay()

    def _restore_runs(self, parked: dict | None):
        """Adopt the previous widget's parked runs, else re-load every
        surviving run dir named in ``roi_runs.json``."""
        if parked is not None:
            manager = parked.get("manager")
            if manager is not None:
                self.manager = manager
            if self._adopted_store:
                self.trace_sets = parked.get("trace_sets") or {}
                self._traces_changed()
                self.derived = [
                    s for s in (parked.get("derived") or [])
                    if tuple(s.result.shape) == (self.ny, self.nx)
                ]
                self._registry_extra = list(parked.get("extra") or [])
                return
            # the parked sets and traces key uids of the previous data's
            # store; fall through to this data's own registry
        if self.fpath is None:
            return
        self._restoring = True
        try:
            for entry in load_run_registry(registry_path(self.fpath)):
                path = Path(entry["path"])
                if not run_dir_complete(path):
                    # a spawned pipeline may have suffixed the dir name
                    hits = sorted(
                        d for d in path.parent.glob(path.name + "*")
                        if d.is_dir() and run_dir_complete(d)
                    )
                    if hits:
                        path = hits[0]
                        entry = {**entry, "path": str(path)}
                if run_dir_complete(path):
                    if self.load_run(path, discarded=entry.get("discarded", ()),
                                     classes=entry.get("classes"),
                                     colors=entry.get("colors")):
                        continue
                self._registry_extra.append(entry)
            own = labels_path(self.fpath).parent
            if run_dir_complete(own) and not any(
                s.result.path == own for s in self.derived
            ):
                # the data sits in a suite2p / masknmf result dir: show its ROIs
                self.load_run(own)
        finally:
            self._restoring = False

    def _autosave(self):
        if self._writer is None:
            return
        try:
            self._writer.save_dirty(self.store, source_path=self.fpath)
            self._save_error = None
        except OSError as e:
            if self._save_error is None:
                self.logger.warning(f"autosave to {self._writer.path} failed: {e}")
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

    def _save_registry(self):
        """Mirror the loaded sets (plus not-yet-loadable entries) into the
        ``roi_runs.json`` sidecar."""
        if self.fpath is None or self._restoring:
            return
        loaded = {str(s.result.path) for s in self.derived}
        entries = [
            {"path": str(s.result.path), "kind": s.result.kind,
             "discarded": s.discarded, "classes": s.classes,
             "colors": {k: list(v) for k, v in s.colors.items()}}
            for s in self.derived
        ]
        entries += [e for e in self._registry_extra if str(e["path"]) not in loaded]
        try:
            save_run_registry(registry_path(self.fpath), entries)
        except OSError as e:
            self._save_error = f"run registry save failed: {e}"

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
    # traces and runs, through the movie view
    # ------------------------------------------------------------------

    def _channel(self) -> int:
        cdim = find_slider_name(self.iw.dim_names, "c")
        return int(self.iw.indices[cdim]) if cdim is not None else 0

    def movie(self, z: int | None = None) -> PlaneMovie | None:
        """``(T, Y, X)`` view of the viewer's array behind store plane ``z``
        (default: the plane on screen), or None when the array cannot be
        wrapped. The plane supplies every scroll dim it encodes (z, channel,
        ...); dims outside the plane key follow the viewer."""
        plane = self.z if z is None else int(z)
        pos = self._plane_pos(plane)
        zz = pos.get(self.zdim, 0) if self.zdim is not None else 0
        cc = pos.get(self.cdim) if self.cdim is not None else 0
        if cc is None:
            cc = self._channel()
        try:
            arr = self.iw.data[0]
            nz = PlaneMovie(arr).nz
            return PlaneMovie(arr, z=(zz if nz > 1 else 0), c=cc)
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    @property
    def trace_busy(self) -> bool:
        self._trace_threads = [t for t in self._trace_threads if t.is_alive()]
        return bool(self._trace_threads)

    @property
    def busy(self) -> bool:
        """anything still working: runs in the manager or trace threads"""
        return self.trace_busy or self.manager.busy

    def has_traces(self) -> bool:
        """anything the Traces tab could plot"""
        return any(ts.data for ts in self.trace_sets.values()) or any(
            s.result.F is not None for s in self.derived
        )

    def trace_disabled(self, index: int) -> str | None:
        movie = self.movie()
        if movie is None or int(movie.shape[0]) < 2:
            return "no (T, Y, X) movie behind this view"
        return None

    def quick_trace(self, index: int):
        """Mean of the ROI's pixels per frame, on a thread tracked as a job."""
        if not 0 <= index < self.n_rois:
            return
        record = self.store.rois[index]
        movie = self.movie(record.z)
        if movie is None:
            return
        uid = record.uid
        mask = self.store.labels[record.z] == (index + 1)
        weights = feather_mask(mask)
        # traces taken at different binnings live on different time bases;
        # record it so the table can say so
        averaged = int(getattr(self.host, "frame_average", 1) or 1)
        description = f"quick trace - ROI {index}"
        job = get_process_manager().start_job("roi_trace", description)
        self.status = f"{description} started"

        def run():
            try:
                y = roi_trace(movie, mask, weights=weights)
            except Exception as error:  # noqa: BLE001 - reported on the job
                self.logger.exception(f"{description} failed")
                job.fail(f"{type(error).__name__}: {error}")
                self._trace_results.put((uid, None, str(error)))
                return
            job.done(f"{y.size} frames")
            entry = {"F": np.asarray(y, np.float32), "frame_average": averaged}
            self._trace_results.put((uid, entry, None))

        thread = threading.Thread(target=run, name=f"roi-trace-{index}", daemon=True)
        self._trace_threads.append(thread)
        thread.start()

    def _traces_changed(self):
        """Trace entries moved: drop stale stats and selections, refit the plot."""
        self._trace_stats.clear()
        self._trace_display.clear()
        self.trace_sel &= set(self._trace_rows())
        self._trace_fit = True

    def _derived_entry(self, res, k: int) -> dict | None:
        """One component's traces straight from a run's arrays."""
        if res.F is None or not 0 <= k < len(res.F):
            return None
        entry = {"F": np.asarray(res.F[k], np.float32)}
        if res.Fneu is not None:
            entry["Fneu"] = np.asarray(res.Fneu[k], np.float32)
        if res.norm is not None:
            entry["norm"] = np.asarray(res.norm[k], np.float32)
        return entry

    def _merge_run_traces(self, res, name: str) -> int:
        """Merge one run's rows into ``trace_sets[name]``, keyed by store
        uid: ``res.uids`` first, else legacy ``store_indices`` mapped
        through the current store. Returns how many rows landed."""
        if res.F is None:
            return 0
        uids = res.uids
        if uids is None and res.store_indices is not None:
            uids = np.full(len(res.stat), -1, np.int64)
            for row, i in enumerate(res.store_indices):
                if 0 <= int(i) < self.n_rois:
                    uids[row] = self.store.rois[int(i)].uid
                else:
                    self.logger.info(
                        f"manual_roi: {name} row {row} maps to missing ROI {i}; skipped"
                    )
        if uids is None:
            return 0
        live = {r.uid for r in self.store.rois}
        ts = self.trace_sets.setdefault(name, TraceSet(name, res.kind))
        merged = 0
        for row, uid in enumerate(uids):
            uid = int(uid)
            if uid not in live:
                continue
            entry = self._derived_entry(res, row)
            if entry is None:
                continue
            ts.data[uid] = entry
            self.trace_uid = uid
            merged += 1
        if not ts.data:
            self.trace_sets.pop(name, None)
        return merged

    def _run_out_dir(self, tag: str) -> Path | None:
        if self.fpath is None:
            self.status = "no data path to write beside"
            return None
        if any(r.tag == tag and r.job is not None and not r.finished for r in self.manager.runs):
            self.status = f"{OUT_PREFIX}{tag} is still being written"
            return None
        return labels_path(self.fpath).parent / f"{OUT_PREFIX}{tag}"

    def _next_find_tag(self) -> str:
        base = labels_path(self.fpath).parent if self.fpath is not None else None
        used = {r.tag for r in self.manager.runs}
        n = 1
        while True:
            tag = f"find{n:02d}"
            if tag not in used and (base is None or not (base / f"{OUT_PREFIX}{tag}").exists()):
                return tag
            n += 1

    def run_rois(self, indices: list[int], tag: str):
        """Send drawn ROIs through ``self.process`` on the viewer's own
        array, writing ``rois_<tag>/`` beside the data. The run closes over
        a store snapshot, so drawing on is safe while it works."""
        indices = [i for i in indices if 0 <= i < self.n_rois]
        if not indices:
            self.status = "nothing to run"
            return
        tag = (tag or "").strip() or "manual"
        out_dir = self._run_out_dir(tag)
        if out_dir is None:
            return
        store = self.store.snapshot()
        arr = self.iw.data[0]
        process = self.process
        c = self._channel()
        planes = sorted({store.rois[i].z for i in indices})
        # a plane keyed by more than z (a channel, say) needs its own movie
        # view; a plain z plane keeps the raw array so ops lookup still works
        multi = any(n != self.zdim for n, _ in self.plane_axes)
        movies = {z: self.movie(z) for z in planes} if multi else {}
        labels = {z: self._plane_label(z) for z in planes}
        logger = self.logger

        def fn(job):
            outs = []
            for i, z in enumerate(planes):
                job.set_progress(i / len(planes), labels[z])
                on_plane = [j for j in indices if store.rois[j].z == z]
                dest = out_dir if len(planes) == 1 else out_dir / f"z{z + 1:02d}"
                src = movies.get(z) or arr
                if process == "demix":
                    out = demix_rois(src, store, on_plane, z=z, c=c, out_dir=dest, tag=tag, logger=logger)
                else:
                    engine = "suite2p" if process == "extract-s2p" else "mean"
                    out = extract_rois(src, store, on_plane, z=z, c=c, out_dir=dest,
                                       engine=engine, tag=tag, logger=logger)
                if out is not None:
                    outs.append(Path(out))
            return outs

        run = RoiRun(
            kind=process, tag=tag,
            description=f"{process}: {len(indices)} ROI(s) -> {out_dir.name}",
            out_root=out_dir, planes=[z + 1 for z in planes],
        )
        self.manager.submit(run, fn, heavy=(process == "demix"))
        self._run_error = None
        self.status = f"{run.description} started"

    def run_roi(self, index: int):
        self.run_rois([index], f"roi{index:04d}")

    def run_in_view(self):
        """Run every drawn ROI the table currently lists."""
        listed = [self.rows[int(r)][1] for r in self.order.order if self.rows[int(r)][0] < 0]
        self.run_rois(listed, self.run_tag)

    def discover_region(self, engine: str):
        """Detect ROIs inside ``self.region`` on the plane on screen; the
        region is consumed by the submit."""
        if self.region is None:
            self.status = "draw a region with r first"
            return
        tag = self._next_find_tag()
        out_dir = self._run_out_dir(tag)
        if out_dir is None:
            return
        box = self.region
        arr = self.iw.data[0]
        z = self.z
        c = self._channel()
        multi = any(n != self.zdim for n, _ in self.plane_axes)
        src = (self.movie(z) or arr) if multi else arr
        logger = self.logger

        def fn(job):
            job.set_progress(0.05, f"{engine} in {box[0]}:{box[1]}, {box[2]}:{box[3]}")
            return discover_rois(src, box, engine=engine, z=z, c=c, out_dir=out_dir, tag=tag, logger=logger)

        run = RoiRun(
            kind="discover", tag=tag,
            description=f"find ({engine}) -> {out_dir.name}",
            out_root=out_dir, box=box, planes=[z + 1],
        )
        self.manager.submit(run, fn, heavy=True)
        self.clear_region()
        self._run_error = None
        self.status = f"{run.description} started"

    def run_full_plane(self, kind: str):
        """Spawn a full suite2p / masknmf run of the plane on screen as a
        detached worker (the Run tab covers full volumes and settings)."""
        if self.fpath is None:
            self.status = "no data path to run on"
            return
        plane = self._plane_pos(self.z).get(self.zdim, 0) + 1
        try:
            args = full_plane_args(kind, self.fpath, plane, self.iw)
        except ValueError as e:
            self.status = str(e)
            return
        run = RoiRun(kind=kind, tag=f"plane{plane:02d}",
                     description=f"{kind} plane{plane:02d}", planes=[plane])
        self.manager.spawn(run, kind, args)
        if run.pid is None:
            return
        run.out_dirs = [Path(args["output_dir"]) / f"zplane{plane:02d}"]
        self._registry_extra.append(
            {"path": str(run.out_dirs[0]), "kind": kind, "discarded": []}
        )
        self._save_registry()
        self.status = f"{run.description} started (pid {run.pid})"

    def _poll_jobs(self):
        """Drain finished traces and runs; called once per frame from the panel."""
        while True:
            try:
                uid, entry, error = self._trace_results.get_nowait()
            except queue.Empty:
                break
            index = self.store.uid_index(uid)
            if error is not None:
                shown = index if index is not None else f"uid {uid}"
                self.status = f"trace for ROI {shown} failed: {error}"
                continue
            if index is None:
                continue  # deleted while the trace ran
            ts = self.trace_sets.setdefault("quick", TraceSet("quick", "quick"))
            ts.data[uid] = entry
            self._traces_changed()
            self.trace_uid = uid
            self.focus_traces = True
            self.status = f"ROI {index}: {entry['F'].size} frames"
        for run, payload in self.manager.poll(get_process_manager()):
            if run.error is not None:
                self._run_error = f"{run.description} failed: {run.error}"
                continue
            if run.kind == "discover" and payload is None:
                self.status = f"{run.description}: nothing found in the region"
                continue
            if run.job is not None:
                if isinstance(payload, (str, Path)):
                    run.out_dirs = [Path(payload)]
                elif payload:
                    run.out_dirs = [Path(o) for o in payload]
                outs = [d for d in run.out_dirs if run_dir_complete(d)]
            else:
                # spawned pipelines may suffix the plane dir name, so
                # resolve the real dirs from disk instead of the guess
                outs = finished_dirs(run.out_root, run.planes) if run.out_root else []
                if outs:
                    guessed = {str(d) for d in run.out_dirs}
                    self._registry_extra = [
                        e for e in self._registry_extra if str(e["path"]) not in guessed
                    ]
                    run.out_dirs = outs
            loaded = sum(self.load_run(d) for d in outs)
            run.loaded = bool(loaded)
            if outs and loaded == len(outs):
                self._run_error = None
                names = ", ".join(d.name for d in outs)
                took = run.job.elapsed_str() if run.job is not None else ""
                self.logger.info(f"roi run done: {names}" + (f" in {took}" if took else ""))
                self.status = f"done: {names}"
                # the run browser is gone: its timing and outputs belong on the
                # job, which the status button and process console already show
                if run.job is not None:
                    run.job.status_message = f"{names} · {took}"
            elif not outs:
                self.status = f"{run.description}: nothing written"
                if run.job is not None:
                    run.job.status_message = "nothing written"

    # ------------------------------------------------------------------
    # keys
    # ------------------------------------------------------------------

    def handle_keys(self):
        io = imgui.get_io()
        if io.want_text_input:
            return
        claim_arrow_keys(("up_arrow", "down_arrow"))
        if imgui.is_key_pressed(imgui.Key.a, False):
            self.set_drawing(not self.drawing)
        if imgui.is_key_pressed(imgui.Key.r, False):
            self.set_region_mode(not self.region_mode)
        if imgui.is_key_pressed(imgui.Key.escape):
            if self.drawer.armed:
                self._arm_mode("off")
            elif self.region is not None:
                self.clear_region()
            elif self.buffer:
                self.buffer_clear()
                self.status = "group cleared"
        if io.key_ctrl and imgui.is_key_pressed(imgui.Key.z, False):
            self.delete_roi(self.n_rois - 1)
        if imgui.is_key_pressed(imgui.Key.delete, False):
            self.delete_selected()
        if imgui.is_key_pressed(imgui.Key.u, False):
            self.next_unlabeled()
        if imgui.is_key_pressed(imgui.Key.f, False):
            self.toggle_follow()
        if imgui.is_key_pressed(imgui.Key.up_arrow):
            self.step(-1)
        if imgui.is_key_pressed(imgui.Key.down_arrow):
            self.step(1)
        if imgui.is_key_pressed(imgui.Key.b, False):
            self.toggle_drawn_overlay()
        if imgui.is_key_pressed(imgui.Key.d, False):
            self.toggle_derived_overlay()
        if imgui.is_key_pressed(imgui.Key.t, False) and self.selected >= 0:
            if self.trace_disabled(self.selected) is None:
                self.quick_trace(self.selected)
        if self.selected_derived is not None:
            if imgui.is_key_pressed(imgui.Key.y, False):
                self.promote_derived(*self.selected_derived)
            if imgui.is_key_pressed(imgui.Key.n, False):
                self.discard_derived(*self.selected_derived, advance=True)
            if imgui.is_key_pressed(imgui.Key.x, False):
                self.set_accepted(*self.selected_derived)
        if self.selected >= 0 or self.selected_derived is not None:
            picked = self.classes.hotkey_pressed()
            if picked is not None:
                self.assign_class(picked)

    # ------------------------------------------------------------------
    # imgui: top panel
    # ------------------------------------------------------------------

    # the top strip's tab selection is the widget's; these keep the old
    # attribute names working for callers and tests
    @property
    def top_tab(self) -> str | None:
        return self.tools_window.active

    @property
    def focus_top(self) -> str | None:
        return self.tools_window._focus

    @focus_top.setter
    def focus_top(self, key: str | None):
        if key is not None:
            self.tools_window.focus(key)

    @property
    def _right_tab_now(self) -> str:
        return self.tools_window._right_now

    @_right_tab_now.setter
    def _right_tab_now(self, name: str):
        self.tools_window.report_right_tab(name)

    @property
    def _focus_right(self) -> str | None:
        return self.tools_window._right_focus

    @_focus_right.setter
    def _focus_right(self, name: str | None):
        self.tools_window._right_focus = name

    def _frame(self):
        """Per-frame work the strip runs whatever tab is on top: background
        jobs, keyboard handling, and our own floating windows."""
        self._poll_jobs()
        self.handle_keys()
        if self.focus_traces:
            self.focus_traces = False
            self.tools_window.focus("traces")
        self.summary.draw()
        self._draw_help_popup()
        self._draw_keybinds_popup()

    def _draw_roi_panel(self):
        """The ROI panel: control cards over a status row, each card gated by
        its Widgets-menu subwidget toggle.

        Cards share the width evenly and never go below ``MIN_CARD_EM``, so
        nothing clips off the right edge: a window too narrow to hold them on
        one row wraps them onto more, and the strip is asked for a row's more
        height to match. Narrower than ``MAX_CARD_ROWS`` rows can show, the
        panel collapses to its placeholder line.
        """
        cards = [self._draw_navigate_card]
        if sub_enabled("manual_roi", "tools"):
            cards.append(self._draw_draw_card)
        if sub_enabled("manual_roi", "overlay"):
            cards.append(self._draw_view_card)
        if sub_enabled("manual_roi", "labels"):
            cards.append(self._draw_labels_card)
        if sub_enabled("manual_roi", "process"):
            cards.append(self._draw_process_card)
        gap = em(0.6)
        min_w = em(MIN_CARD_EM)
        fewest = -(-len(cards) // MAX_CARD_ROWS)
        with fit_width(
            "ROI tools", min_width=fewest * min_w + (fewest - 1) * gap
        ) as shown:
            if not shown:
                self._roi_panel.height = PANEL_HEIGHT
                return
            avail = imgui.get_content_region_avail().x
            per_row, rows, w = card_grid(len(cards), avail, min_w, gap)
            self._roi_panel.height = PANEL_HEIGHT * rows
            vgap = imgui.get_style().item_spacing.y
            body = max(imgui.get_content_region_avail().y - em(1.8), em(6))
            h = max((body - vgap * (rows - 1)) / rows, em(6))
            for i, draw in enumerate(cards):
                if i % per_row:
                    imgui.same_line(0, gap)
                draw(h, w)
            self._draw_status()

    def _draw_navigate_card(self, h: float, w: float = 0.0):
        with card("##nav", "NAVIGATE", h, w):
            if imgui.button("prev"):
                self.step(-1)
            imgui.same_line(0, em(0.4))
            if imgui.button("next"):
                self.step(1)
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(up / down)")
            n = len(self.order.order)
            imgui.text(f"{self.order.pos + 1 if n else 0} / {n} in view")
            if imgui.button("next unlabeled"):
                self.next_unlabeled()
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(u)")
            changed, self.follow = imgui.checkbox("center & advance", self.follow)
            if changed and self.follow:
                self._center_selection()
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(f)")
            if imgui.button("Open full FOV"):
                self.open_full_fov()

    def _draw_draw_card(self, h: float, w: float = 0.0):
        with card("##draw", "DRAW", h, w):
            with selected_button_style(self.drawing):
                if imgui.button("Add ROI"):
                    self.set_drawing(not self.drawing)
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(a)")
            imgui.same_line(0, em(0.6))
            with selected_button_style(self.region_mode):
                if imgui.button("Region"):
                    self.set_region_mode(not self.region_mode)
            if imgui.is_item_hovered():
                imgui.set_tooltip("drag a box; discovery runs inside it")
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(r)")
            if imgui.button("Undo"):
                self.delete_roi(self.n_rois - 1)
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(ctrl+z)")
            imgui.same_line(0, em(0.6))
            nothing = self.selected < 0 and self.selected_derived is None
            if nothing:
                imgui.begin_disabled()
            if imgui.button("Discard" if self.selected_derived is not None else "Delete"):
                self.delete_selected()
            if nothing:
                imgui.end_disabled()
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(del)")
            imgui.same_line(0, em(0.6))
            if imgui.button("Clear"):
                self.clear()
            imgui.text_disabled(f"{self.n_rois} ROIs")
            if self.region is not None:
                y0, y1, x0, x1 = self.region
                imgui.text_disabled(f"region {y1 - y0}x{x1 - x0}")
                imgui.same_line(0, em(0.4))
                if imgui.small_button("clear##region"):
                    self.clear_region()

    def _draw_view_card(self, h: float, w: float = 0.0):
        with card("##view", "VIEW", h, w):
            dirty = False
            changed, self.show_masks = imgui.checkbox("drawn", self.show_masks)
            dirty |= changed
            set_tooltip("The ROIs you drew by hand", show_mark=False)
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(b)")
            imgui.same_line(0, em(0.6))
            imgui.set_next_item_width(em(6))
            changed, self.opacity = imgui.slider_float("##opacity", self.opacity, 0.05, 1.0, "opacity %.2f")
            dirty |= changed
            if dirty:
                self.refresh_overlay()
            dirty = False
            changed, self.show_derived = imgui.checkbox("algo", self.show_derived)
            dirty |= changed
            set_tooltip("ROIs an algorithm found (find / demix / full-plane runs)", show_mark=False)
            imgui.same_line(0, em(0.4))
            imgui.text_disabled("(d)")
            imgui.same_line(0, em(0.6))
            imgui.set_next_item_width(em(6))
            changed, self.derived_opacity = imgui.slider_float(
                "##derived_opacity", self.derived_opacity, 0.05, 1.0, "opacity %.2f"
            )
            dirty |= changed
            if dirty:
                self.refresh_derived_overlay()
            if imgui.button("Save"):
                self.save()
            imgui.same_line(0, em(0.6))
            self._draw_save_note()

    def _draw_labels_card(self, h: float, w: float = 0.0):
        with card("##labels", "LABELS", h, w):
            if self.n_rois:
                # just the count: NAVIGATE already carries "next unlabeled (u)"
                done = int((self.classes.labels[: self.n_rois] >= 0).sum())
                imgui.text_colored(
                    to_vec4(THEME.ok if done == self.n_rois else THEME.warn),
                    f"labeled {done}/{self.n_rois}",
                )
            self.new_label, changed = draw_label_editor(self.classes, self.new_label, "_roi")
            if changed:
                self._sync_store_from_classes()
                self.order.rebuild()
                self.refresh_overlay()
                self._autosave()
            imgui.dummy(imgui.ImVec2(0, 0))
            picked = self._draw_label_columns()
            if picked == UNLABEL_ALL:
                self.unlabel_all()
            elif picked is not None:
                self.assign_class(picked)

    def _draw_label_columns(self):
        """One button per class, split into two columns filled evenly; more
        columns only when the card's height cannot hold half the entries.

        Buttons are sized to their column rather than to their text, so a
        long class name widens nothing and the columns stay even.
        """
        picked = None
        row_h = imgui.get_frame_height_with_spacing()
        rows = max(int(imgui.get_content_region_avail().y // row_h), 1)
        entries: list[tuple[str, int | None]] = [
            ("label", i) for i in range(len(self.classes.names))
        ]
        if self.classes.names:
            entries += [("unlabel", None), ("unlabel_all", None)]
        if not entries:
            return None
        ncols = max(2, -(-len(entries) // rows))
        per_col = -(-len(entries) // ncols)
        gap, hint = em(0.8), em(2.0)
        col_w = max(
            (imgui.get_content_region_avail().x - gap * (ncols - 1)) / ncols, em(5)
        )
        size = imgui.ImVec2(max(col_w - hint, em(3.5)), 0)
        for c0 in range(0, len(entries), per_col):
            if c0:
                imgui.same_line(0, gap)
            imgui.begin_group()
            for kind, i in entries[c0 : c0 + per_col]:
                if kind == "label":
                    with label_button(self.classes.color(i)):
                        if imgui.button(
                            f"{self.classes.names[i]} ({self.classes.count(i)})##lab{i}",
                            size,
                        ):
                            picked = i
                    if i < 9:
                        imgui.same_line(0, 4)
                        imgui.text_disabled(f"({i + 1})")
                elif kind == "unlabel":
                    if imgui.button("unlabel##_roi", size):
                        picked = UNLABELED
                    imgui.same_line(0, 4)
                    imgui.text_disabled("(0)")
                else:
                    with danger_button():
                        if imgui.button("unlabel all##_roi", size):
                            picked = UNLABEL_ALL
            imgui.end_group()
        return picked

    def _caption(self, text: str, width: float):
        """Dim row caption at a fixed width, so the rows line up."""
        imgui.align_text_to_frame_padding()
        imgui.text_disabled(text)
        imgui.same_line(width)

    def _draw_process_card(self, h: float, w: float = 0.0):
        """One action per row, each with a dim caption saying what it does:
        run the drawn ROIs, find new ROIs in the region, run a whole plane."""
        with card("##process", "PROCESS", h, w):
            cap = em(3.6)
            self._caption("ROIs", cap)
            imgui.set_next_item_width(em(7))
            changed, sel = imgui.combo("##process", PROCESSES.index(self.process), list(PROCESSES))
            if changed:
                self.process = PROCESSES[sel]
            set_tooltip(
                "What to run on the drawn ROIs:\n"
                "extract - mean traces from the masks\n"
                "extract-s2p - suite2p's own extractor\n"
                "demix - masknmf NMF seeded with the masks",
                show_mark=False,
            )
            imgui.same_line(0, em(0.4))
            imgui.set_next_item_width(em(4.5))
            _, self.run_tag = imgui.input_text_with_hint("##run_tag", "tag", self.run_tag)
            set_tooltip("Names the output folder: rois_<tag>/ beside the data", show_mark=False)
            imgui.same_line(0, em(0.4))
            if imgui.button("Run"):
                self.run_in_view()
            set_tooltip(
                "Run every drawn ROI listed in the table through the picked "
                "process; outputs land in rois_<tag>/ beside the data",
                show_mark=False,
            )
            self._caption("find", cap)
            no_region = self.region is None
            if no_region:
                imgui.begin_disabled()
            find_masknmf = imgui.button("masknmf##find")
            hovered = imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
            imgui.same_line(0, em(0.4))
            find_suite2p = imgui.button("suite2p##find")
            hovered |= imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled)
            if no_region:
                imgui.end_disabled()
            if hovered:
                imgui.set_tooltip(
                    "Detect new ROIs inside a region, unseeded - draw the "
                    "region with r first" if no_region
                    else "Detect new ROIs inside the region, unseeded; they "
                         "arrive as an algo overlay to promote or discard"
                )
            if find_masknmf:
                self.discover_region("masknmf")
            if find_suite2p:
                self.discover_region("suite2p")
            self._caption("plane", cap)
            no_path = self.fpath is None
            if no_path:
                imgui.begin_disabled()
            for i, kind in enumerate(("suite2p", "masknmf")):
                if i:
                    imgui.same_line(0, em(0.4))
                if imgui.button(f"{kind}##plane"):
                    self.run_full_plane(kind)
                if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
                    imgui.set_tooltip(
                        "open a file first" if no_path else
                        f"Full {kind} run of the plane on screen -> "
                        f"zplane{self._plane_pos(self.z).get(self.zdim, 0) + 1:02d}/ "
                        "beside the data.\nRun tab for full volume / settings."
                    )
            if no_path:
                imgui.end_disabled()

    def _status_message(self) -> tuple[tuple, str]:
        if self._save_error is not None:
            return THEME.err, self._save_error
        if self._run_error is not None:
            return THEME.err, self._run_error
        active = self.manager.active
        if active:
            verbs = {"discover": "find", "extract-s2p": "extract"}
            names = ", ".join(
                f"{verbs.get(r.kind, r.kind)} "
                + (f"{OUT_PREFIX}{r.tag}" if r.job is not None else r.tag)
                for r in active
            )
            return THEME.warn, f"{len(active)} running: {names}"
        return THEME.text_dim, self.status

    def _draw_status(self):
        """The status message with right-aligned counts. Help / keybinds live
        on the menu row, beside the Metadata Viewer button."""
        color, text = self._status_message()
        imgui.align_text_to_frame_padding()
        imgui.text_colored(to_vec4(color), text)
        imgui.same_line(0, em(1.0))
        avail = imgui.get_content_region_avail().x
        if avail <= em(1):
            return
        m = sum(len(s.result.stat) - len(s.discarded) for s in self.derived)
        counts = f"{self.n_rois} drawn · {m} algo"
        with imgui_ctx.begin_child(
            "##roi_counts", imgui.ImVec2(avail, em(1.6)),
            imgui.ChildFlags_.none, imgui.WindowFlags_.no_scrollbar,
        ):
            width = imgui.calc_text_size(counts).x
            inner = imgui.get_content_region_avail().x
            if width < inner:
                imgui.set_cursor_pos_x(inner - width)
            imgui.align_text_to_frame_padding()
            imgui.text_disabled(counts)

    def _draw_save_note(self):
        if self._writer is None:
            imgui.text_disabled("autosave off")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "ROIs are kept in memory only - open a file to autosave "
                    "beside it, or press Save"
                )
            return
        imgui.text_disabled("Autosaved")
        if not imgui.is_item_hovered():
            return
        imgui.begin_tooltip()
        imgui.text(f"labels zarr: {self._save_target()}")
        imgui.text_colored(
            to_vec4(THEME.code),
            "from mbo_utilities.annotation import LabelsZarr\n"
            f'store = LabelsZarr.load(r"{self._save_target()}")\n'
            "store.labels        # (Z, Y, X) uint16; 0 = bg, ROI i = i + 1\n"
            "store.rois          # per-ROI plane, area, class, note, uid, source",
        )
        imgui.end_tooltip()

    def _draw_help_popup(self):
        if not self.help_open:
            return
        opened, self.help_open = popup("ROI help", self.help_open)
        if opened:
            section("Workflow")
            for i, step in enumerate(_HELP_STEPS, 1):
                imgui.text_colored(to_vec4(THEME.accent), f"{i}.")
                imgui.same_line(em(2.6))
                imgui.text(textwrap.fill(step, 80))
                imgui.dummy(imgui.ImVec2(0, em(0.15)))
            section("Output files")
            imgui.text_colored(to_vec4(THEME.code), _HELP_FILES)
            if close_button():
                self.help_open = False
        imgui.end()

    def _draw_keybinds_popup(self):
        if not self.keybinds_open:
            return
        opened, self.keybinds_open = popup("ROI keys", self.keybinds_open)
        if opened:
            flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_h
            if imgui.begin_table("##roi-keybinds", 2, flags):
                imgui.table_setup_column("key", imgui.TableColumnFlags_.width_fixed, em(10))
                imgui.table_setup_column("action")
                for key, action in KEYBINDS:
                    imgui.table_next_row()
                    imgui.table_next_column()
                    imgui.text_colored(to_vec4(THEME.warn), key)
                    imgui.table_next_column()
                    imgui.text(action)
                imgui.end_table()
            if close_button():
                self.keybinds_open = False
        imgui.end()

    # ------------------------------------------------------------------
    # imgui: right tabs
    # ------------------------------------------------------------------

    def draw_tab(self):
        """The ROIs tab: filters, the combined drawn + derived table, and
        the selection footer."""
        changed_any = False
        if self.store.nz > 1:
            on = self.order.plane is not None
            changed, on = imgui.checkbox(
                f"this plane ({self._plane_label(self.z)} of {self.store.nz})", on
            )
            if changed:
                self.order.plane = self.z if on else None
                changed_any = True
        half = max((imgui.get_content_region_avail().x - em(0.5)) / 2, em(6))
        if draw_label_filter(self.order, self.classes, "_roi", half):
            changed_any = True
        set_tooltip("Filter by label", show_mark=False)
        imgui.same_line(0, em(0.5))
        names = ["all", "drawn", *(s.name for s in self.derived)]
        current = 0 if self.order.source is None else self.order.source + 1
        imgui.set_next_item_width(-1)
        changed, sel = imgui.combo("##source_filter", min(current, len(names) - 1), names)
        set_tooltip("Filter by source: drawn by hand, or a loaded run", show_mark=False)
        if changed:
            self.order.source = None if sel == 0 else sel - 1
            changed_any = True
        if draw_range_filter(self.order, "_roi"):
            changed_any = True
        imgui.text(f"{len(self.order.order)}/{self.order.n_items} in view")
        if changed_any:
            self.order.rebuild()

        footer = 2 * imgui.get_frame_height_with_spacing() + 12
        with imgui_ctx.begin_child("##roi_table", imgui.ImVec2(0, -footer)):
            if self.rows:
                pos = self.order.pos
                self.scroll_to_selection = draw_roi_table(
                    self.order,
                    self.classes,
                    self.columns,
                    self._formatters(),
                    self.scroll_to_selection,
                    table_id="manual_rois",
                    on_select=self._table_select,
                    actions=self.row_actions,
                    is_grouped=self._row_grouped,
                    on_ctrl_select=self._table_ctrl,
                    on_shift_select=self.buffer_extend_to,
                )
                if self.order.pos != pos and self.order.current is not None:
                    self.select_row(self.order.current)
            else:
                imgui.text_disabled("no ROIs yet")

        pending, self._pending_row_action = self._pending_row_action, None
        if pending is not None:
            _act, si, k = pending
            if si < 0:
                self.delete_roi(k)
            else:
                self.discard_derived(si, k, advance=self.selected_derived == (si, k))

        imgui.separator()
        if len(self.buffer) > 1:
            imgui.text_disabled(f"{len(self.buffer)} ROIs grouped")
            imgui.same_line(0, 8)
            changed, col = imgui.color_edit3(
                "##group_color", list(self._group_color),
                imgui.ColorEditFlags_.no_inputs,
            )
            if changed:
                self._group_color = tuple(col)
            imgui.same_line(0, 4)
            if imgui.small_button("color group"):
                self.set_group_color(self._group_color)
            set_tooltip("Give every grouped ROI this color", show_mark=False)
            imgui.same_line(0, 4)
            if imgui.small_button("reset color"):
                self.set_group_color(None)
            set_tooltip("Back to class / hue colors", show_mark=False)
            imgui.same_line(0, 4)
            if imgui.small_button("ungroup"):
                self.buffer_clear()
            set_tooltip("Empty the group (esc)", show_mark=False)
            imgui.text_disabled("label buttons and keys 1-9 apply to the whole group")
        elif self.selected >= 0:
            imgui.set_next_item_width(-1)
            changed, self._note_buf = imgui.input_text_with_hint("##note", "note", self._note_buf)
            if changed:
                self.store.set_note(self.selected, self._note_buf)
            if imgui.is_item_deactivated_after_edit():
                self._autosave()
            if imgui.button("Delete selected"):
                self.delete_roi(self.selected)
        elif self.selected_derived is not None:
            si, k = self.selected_derived
            s = self.derived[si]
            promoted = (s.name, k) in self._promoted
            imgui.text_disabled(f"{s.name} row {k}" + (" · promoted" if promoted else ""))
            if promoted:
                imgui.begin_disabled()
            if imgui.button("Promote"):
                self.promote_derived(si, k)
            if promoted:
                imgui.end_disabled()
            imgui.same_line(0, 8)
            if imgui.button("Discard"):
                self.discard_derived(si, k, advance=True)
            imgui.same_line(0, 8)
            if imgui.button("Reject" if s.accepted[k] else "Accept"):
                self.set_accepted(si, k)
        else:
            imgui.text_disabled("select an ROI to note")
            imgui.begin_disabled()
            imgui.button("Delete selected")
            imgui.end_disabled()

    # row actions: callbacks take a table row index and route per kind

    def _act_run(self, row: int):
        si, k = self.rows[row]
        if si < 0:
            self.run_roi(k)

    def _run_disabled(self, row: int) -> str | None:
        return "promote first" if self.rows[row][0] >= 0 else None

    def _act_trace(self, row: int):
        si, k = self.rows[row]
        if si < 0:
            self.quick_trace(k)

    def _trace_row_disabled(self, row: int) -> str | None:
        si, k = self.rows[row]
        if si >= 0:
            return "promote first"
        return self.trace_disabled(k)

    def _act_remove(self, row: int):
        # mutating mid-table-draw rebuilds the rows the clipper is still
        # iterating; run it once draw_roi_table has returned
        si, k = self.rows[row]
        self._pending_row_action = ("remove", si, k)

    @property
    def row_actions(self) -> tuple[RowAction, ...]:
        return (
            RowAction(RUN_ICON, f"Run - {self.process} this ROI", self._act_run, self._run_disabled),
            RowAction(TRACE_ICON, "Quick trace - mean of this ROI per frame", self._act_trace, self._trace_row_disabled),
            RowAction(REMOVE_ICON, "Remove - delete the drawn ROI, discard the algo one", self._act_remove),
        )

    # ------------------------------------------------------------------
    # imgui: traces tab
    # ------------------------------------------------------------------

    def _lines_for_uid(self, uid):
        """``(header, [(label, key), ...])`` for one ROI uid, or None."""
        lines = [
            (name, ("uid", name, uid))
            for name, ts in self.trace_sets.items()
            if ts.visible and uid in ts.data
        ]
        if not lines:
            return None
        index = self.store.uid_index(uid)
        header = f"ROI {index}" if index is not None else f"uid {uid}"
        return header, lines

    def _selection_trace_keys(self) -> list[tuple]:
        """Trace-table keys of the selection; empty when it has none."""
        if self.selected_derived is not None:
            si, k = self.selected_derived
            s = self.derived[si]
            if self._derived_entry(s.result, k) is None:
                return []
            return [("row", s.name, k)]
        if self.selected < 0:
            return []
        got = self._lines_for_uid(self.store.rois[self.selected].uid)
        return [] if got is None else [key for _label, key in got[1]]

    def _sync_trace_sel(self):
        """Point the plotted set at the selection, so the trace in the top
        panel is the ROI the image is showing. A multi-row (ctrl+click)
        selection that already covers it is left alone."""
        keys = set(self._selection_trace_keys())
        if keys and self.trace_sel & keys:
            return
        if self.trace_sel != keys:
            self.trace_sel = keys
            self._trace_fit = True

    def _trace_target(self):
        """``(header, [(label, key), ...])`` for the shown ROI, or None.

        With something selected this is that ROI's traces and nothing else:
        the plot must never show an ROI the image is not showing. Only with
        no selection does it fall back to the last trace collected, then to
        any set that has one.
        """
        if self.selected_derived is not None or self.selected >= 0:
            keys = self._selection_trace_keys()
            if not keys:
                return None
            if self.selected_derived is not None:
                si, k = self.selected_derived
                return f"{self.derived[si].name} row {k}", [(self.derived[si].name, keys[0])]
            return self._lines_for_uid(self.store.rois[self.selected].uid)
        candidates = [self.trace_uid]
        for ts in self.trace_sets.values():
            if ts.data:
                candidates.append(next(iter(ts.data)))
        for uid in candidates:
            got = self._lines_for_uid(uid)
            if got is not None:
                return got
        return None

    def _binning_tag(self, key) -> str:
        """`` x10`` when a trace was taken at a different frame averaging than
        the data now shows — those traces are on another time base."""
        entry = self._trace_entry(key)
        taken = int((entry or {}).get("frame_average", 1) or 1)
        now = int(getattr(self.host, "frame_average", 1) or 1)
        return f" x{taken}" if taken != now and taken > 1 else ""

    def _window_spec(self) -> tuple[str, int]:
        """``(projection, size)`` of the viewer's window function.

        The preview trace gets the same window the image does, so what the
        plot shows is what the frame on screen shows.
        """
        host = self.host
        if host is None:
            return "mean", 1
        try:
            return str(host.proj), max(1, int(host.window_size))
        except Exception:
            return "mean", 1

    def _windowed(self, y):
        """``y`` under the viewer's rolling window; the raw array at size 1."""
        proj, size = self._window_spec()
        if y is None or size <= 1 or y.size < size:
            return y
        cached = self._trace_window_cache.get((id(y), proj, size))
        if cached is not None:
            return cached
        pad = (size - 1) // 2, size // 2
        padded = np.pad(y, pad, mode="edge")
        view = np.lib.stride_tricks.sliding_window_view(padded, size)
        func = {"max": np.max, "std": np.std}.get(proj, np.mean)
        out = np.ascontiguousarray(func(view, axis=-1), np.float32)
        self._trace_window_cache = {(id(y), proj, size): out}
        return out

    def _display(self, key) -> tuple:
        """Cached ``(trace, neuropil)`` display arrays for one trace key."""
        got = self._trace_display.get(key)
        if got is None:
            entry = self._trace_entry(key)
            if entry is None:
                return None, None
            y = np.ascontiguousarray(
                display_trace(entry, self.correct_neuropil), np.float32
            )
            yneu = display_fneu(entry)
            if yneu is not None:
                yneu = np.ascontiguousarray(yneu, np.float32)
            got = (y, yneu)
            self._trace_display[key] = got
        return got

    def _set_by_name(self, name: str):
        for si, s in enumerate(self.derived):
            if s.name == name:
                return si, s
        return None

    def _trace_entry(self, key) -> dict | None:
        """The ``{"F", ...}`` entry behind one trace-table key."""
        origin, name, k = key
        if origin == "uid":
            ts = self.trace_sets.get(name)
            return None if ts is None else ts.data.get(k)
        hit = self._set_by_name(name)
        return None if hit is None else self._derived_entry(hit[1].result, k)

    def _key_to_pair(self, key) -> tuple[int, int] | None:
        """``(si, k)`` behind one trace key, or None when the ROI is gone."""
        origin, name, k = key
        if origin == "uid":
            index = self.store.uid_index(k)
            return (-1, index) if index is not None else None
        hit = self._set_by_name(name)
        return (hit[0], k) if hit is not None else None

    def _trace_color(self, key) -> tuple[float, float, float] | None:
        """The mask color of the ROI behind one trace key, so plot lines and
        table rows match the overlay."""
        pair = self._key_to_pair(key)
        if pair is None:
            return None
        si, k = pair
        if si < 0:
            return tuple(v / 255.0 for v in self.store.roi_rgb(k))
        return component_color(self.derived[si], k)

    def _trace_shown(self, key) -> tuple[float, str]:
        """``(sort value, display text)`` for a key's roi column."""
        origin, name, k = key
        if origin == "uid":
            index = self.store.uid_index(k)
            if index is not None:
                return float(index), f"{index}"
            return float((1 << 30) + k), f"uid {k}"
        index = self._promoted.get((name, k))
        if index is not None:
            return float(index), f"{index}"
        return float((1 << 30) + k), f"{k}"

    def _plot_lines(self):
        """``(header, [(label, key), ...])``: the checked trace-table rows,
        else whatever the current selection points at.

        When the checked rows are exactly the selection's own traces — which
        is what selecting an ROI leaves behind — the plot is titled after the
        ROI rather than counted, so it reads as "this is what the image is
        showing".
        """
        if self.trace_sel and self.trace_sel != set(self._selection_trace_keys()):
            lines = []
            for key in sorted(self.trace_sel):
                if self._trace_entry(key) is None:
                    continue
                _v, shown = self._trace_shown(key)
                lines.append((f"{key[1]} · {shown}", key))
            if lines:
                return f"{len(lines)} selected", lines
        return self._trace_target()

    def draw_traces(self):
        """The Traces tab: the trace-table selection (else the shown ROI)
        as pannable, zoomable lines, the cursor bound to the viewer's t."""
        target = self._plot_lines()
        if target is None:
            imgui.text_disabled(
                f"No traces yet. Use {TRACE_ICON} on a row of the ROIs tab, or run a process."
            )
            return
        header, lines = target
        changed, self.correct_neuropil = imgui.checkbox(
            "neuropil corrected", self.correct_neuropil
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip("subtract 0.7 x Fneu before dF/F (suite2p results)")
        if changed:
            self._trace_display.clear()
            self._trace_stats.clear()
            self._trace_fit = True
        imgui.same_line(0, 14)
        proj, size = self._window_spec()
        window = f" · {proj} {size}" if size > 1 else ""
        imgui.text_disabled(
            f"{header}, frame {self.current_frame()}{window} · "
            "drag pans, scroll zooms, double-click fits"
        )
        height = max(imgui.get_content_region_avail().y - 4, 60.0)
        if implot.get_current_context() is None:
            implot.create_context()
        key = tuple(label for label, _ in lines)
        if key != self._plot_key:
            self._plot_key = key
            self._trace_fit = True
        if self._trace_fit:
            implot.set_next_axes_to_fit()
            self._trace_fit = False
        flags = implot.Flags_.no_title
        if len(lines) <= 1:
            flags |= implot.Flags_.no_legend
        if not implot.begin_plot("##roi_trace_plot", imgui.ImVec2(-1, height), flags):
            return
        try:
            # no auto-fit flags: the axes stay interactive between refits.
            # each line takes its ROI's mask color via a single-color
            # colormap; neuropil is always the same blue
            implot.setup_axes("frame", "dF/F (%)")
            ctrl = imgui.get_io().key_ctrl
            for label, tkey in lines:
                y, yneu = self._display(tkey)
                if y is None:
                    continue
                rgb = self._trace_color(tkey)
                if rgb is not None:
                    implot.push_colormap(_line_colormap(rgb))
                implot.plot_line(label, self._windowed(y))
                if rgb is not None:
                    implot.pop_colormap()
                if yneu is not None:
                    implot.push_colormap(_fneu_colormap())
                    implot.plot_line(f"{label} Fneu", self._windowed(yneu))
                    implot.pop_colormap()
                pair = self._key_to_pair(tkey)
                if pair is not None:
                    # ctrl+click a legend entry: toggle its ROI in the group
                    if ctrl and implot.is_legend_entry_hovered(label) and imgui.is_mouse_clicked(0):
                        self.buffer_toggle(*pair)
                    if implot.begin_legend_popup(label):
                        in_group = pair in self.buffer
                        if imgui.menu_item_simple(
                            "remove from group" if in_group else "add to group"
                        ):
                            self.buffer_toggle(*pair)
                        if imgui.menu_item_simple("select this ROI"):
                            self.buffer_clear()
                            if pair[0] < 0:
                                self.select_roi(pair[1])
                            else:
                                self.select_derived(*pair)
                        implot.end_legend_popup()
            if self.tdim is not None:
                moved, frame = implot.drag_line_x(0, float(self.current_frame()), _CURSOR_COLOR, 1.5)[:2]
                if moved:
                    self.set_frame(round(frame))
        finally:
            implot.end_plot()

    def _sorted_trace_rows(self) -> list[tuple]:
        """Trace-table keys in the order the table shows them, so stepping
        with the arrows walks what the user sees."""
        rows = self._trace_rows()
        col, ascending = self._trace_sort

        def sort_key(key):
            return (self._trace_shown(key)[0], key[1], *self._trace_stat(key))[col]

        rows.sort(key=sort_key, reverse=not ascending)
        return rows

    def _trace_rows(self) -> list[tuple]:
        """``(origin, name, key)`` per listable trace: collected uid-keyed
        entries (quick traces, extract runs), plus every non-discarded
        component of a loaded set that carries traces."""
        derived_names = {s.name for s in self.derived}
        rows = [
            ("uid", name, uid)
            for name, ts in self.trace_sets.items()
            if name not in derived_names
            for uid in ts.data
        ]
        for s in self.derived:
            if s.result.F is None:
                continue
            rows += [
                ("row", s.name, k)
                for k in range(len(s.result.stat))
                if k not in s.discarded
            ]
        return rows

    def _trace_stat(self, key) -> tuple[int, float, float, float]:
        """``(frames, mean, peak, snr)`` of the displayed (dF/F) trace,
        cached until the trace sets change; snr is peak over baseline in
        robust sd units."""
        got = self._trace_stats.get(key)
        if got is None:
            y, _ = self._display(key)
            f = y if y is not None else np.zeros(0, np.float32)
            if f.size:
                med = float(np.median(f))
                mad = float(np.median(np.abs(f - med)))
                peak = float(f.max())
                snr = (peak - med) / (1.4826 * mad) if mad > 0 else 0.0
                got = (int(f.size), float(f.mean()), peak, snr)
            else:
                got = (0, 0.0, 0.0, 0.0)
            self._trace_stats[key] = got
        return got

    def draw_trace_table(self):
        """The right bar's Traces tab: every collected trace with stats;
        click selects one, ctrl+click several — the selection is what the
        top panel plots."""
        rows = self._sorted_trace_rows()
        if not rows:
            imgui.text_disabled(
                f"No traces yet. Use {TRACE_ICON} on a row of the ROIs tab, or run a process."
            )
            return
        imgui.text_disabled(f"{len(rows)} traces · {len(self.trace_sel)} plotted")
        imgui.same_line(0, 12)
        if imgui.small_button("plot all"):
            self.trace_sel = set(rows)
            self._trace_fit = True
        imgui.same_line(0, 6)
        if imgui.small_button("clear"):
            self.trace_sel.clear()
            self._trace_fit = True
        # stretch, not fit-to-content: the tab is a narrow column and
        # fixed-width columns ran off its right edge. frames and peak start
        # hidden — right-click the header to bring them back.
        flags = (
            imgui.TableFlags_.sortable | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.borders_inner_h | imgui.TableFlags_.scroll_y
            | imgui.TableFlags_.resizable | imgui.TableFlags_.hideable
            | imgui.TableFlags_.sizing_stretch_prop
        )
        avail = imgui.get_content_region_avail()
        if not imgui.begin_table("##trace_table", len(TRACE_COLUMNS), flags, imgui.ImVec2(0, avail.y)):
            return
        imgui.table_setup_scroll_freeze(0, 1)
        stretch = imgui.TableColumnFlags_.width_stretch
        for i, (name, weight, hidden) in enumerate(TRACE_COLUMNS):
            column_flags = stretch
            if i == 0:
                column_flags |= imgui.TableColumnFlags_.default_sort
            if hidden:
                column_flags |= imgui.TableColumnFlags_.default_hide
            imgui.table_setup_column(name, column_flags, weight)
        imgui.table_headers_row()
        set_tooltip("Right-click a header to show or hide columns", show_mark=False)
        specs = imgui.table_get_sort_specs()
        if specs is not None and specs.specs_dirty:
            if specs.specs_count > 0:
                self._trace_sort = (
                    int(specs.specs.column_index),
                    specs.specs.sort_direction == imgui.SortDirection.ascending,
                )
            specs.specs_dirty = False
        ctrl = imgui.get_io().key_ctrl
        for key in rows:
            origin, name, k = key
            imgui.table_next_row()
            imgui.table_next_column()
            _v, shown = self._trace_shown(key)
            picked = key in self.trace_sel
            rgb = self._trace_color(key)
            if rgb is not None:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*rgb, 1.0))
            clicked, _ = imgui.selectable(
                f"{shown}##tr_{origin}_{name}_{k}", picked,
                imgui.SelectableFlags_.span_all_columns,
            )
            if rgb is not None:
                imgui.pop_style_color()
            if clicked:
                if ctrl:
                    self.toggle_trace(key)
                else:
                    self.select_trace(key)
            n, mean, peak, snr = self._trace_stat(key)
            source = name + self._binning_tag(key)
            for text in (source, f"{n}", f"{mean:.1f}", f"{peak:.1f}", f"{snr:.1f}"):
                if imgui.table_next_column():
                    imgui.text(text)
        imgui.end_table()

def attach_roi_widget(parent: Any, focus: bool = False) -> ManualRoiWidget | None:
    """Turn the ROI widget on for a ``PreviewDataWidget``; ROIs and runs from
    an earlier toggle this session are adopted. Returns None (logged) when it
    cannot be built."""
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
            runs=getattr(parent, "_manual_roi_runs", None),
            strip=getattr(parent, "top_strip", None),
            host=parent,
        )
    except Exception:
        parent.logger.warning("manual ROI widget unavailable", exc_info=True)
        parent.manual_roi = None
        return None
    widget.focus_tab = focus
    parent.manual_roi = widget
    return widget


def detach_roi_widget(parent: Any) -> None:
    """Turn the ROI widget off, keeping its store and runs for the next toggle."""
    widget = getattr(parent, "manual_roi", None)
    if widget is None:
        return
    parent._manual_roi_store = widget.store
    parent._manual_roi_runs = widget.park_runs()
    widget.close()
    parent.manual_roi = None
