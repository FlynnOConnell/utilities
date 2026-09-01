"""
PreviewDataWidget - Main GUI widget for data preview and processing.

This module contains the PreviewDataWidget class which provides:
- Time series data visualization
- Z-stats signal quality analysis
- Suite2p pipeline integration
- File format conversion

The widget uses modular components:
- _menu_bar.py: Menu bar and status indicator
- _popups.py: Tool popups and process console
- _save_as.py: Save As dialog
- _keyboard.py: Keyboard shortcuts
- _dialogs.py: File dialog handling
- _stats.py: Z-stats computation and display
"""

import logging
import threading
from pathlib import Path
from typing import Literal
import os
import importlib.util
import time

# Force rendercanvas to use Qt backend if PyQt6 is available
# This must happen BEFORE importing fastplotlib to avoid glfw selection
if importlib.util.find_spec("PyQt6") is not None:
    os.environ.setdefault("RENDERCANVAS_BACKEND", "qt")
    import PyQt6  # noqa: F401 - Must be imported before rendercanvas.qt can load

    # Fix suite2p PyQt6 compatibility - must happen before any suite2p GUI imports
    from PyQt6.QtWidgets import QSlider
    if not hasattr(QSlider, "NoTicks"):
        QSlider.NoTicks = QSlider.TickPosition.NoTicks

import imgui_bundle
import numpy as np
from numpy import ndarray
from scipy.ndimage import gaussian_filter

from imgui_bundle import imgui, hello_imgui, imgui_ctx, implot

from mbo_utilities.preferences import get_mbo_dirs
from mbo_utilities.reader import MBO_AVAILABLE_FTYPES
from mbo_utilities.arrays.features import PhaseCorrectionFeature
from mbo_utilities.preferences import get_last_dir
from mbo_utilities.arrays import ScanImageArray
from mbo_utilities.gui._availability import HAS_SUITE2P
from mbo_utilities.gui._imgui_helpers import push_font_safe
from mbo_utilities.gui.widgets.gui_logger import GuiLogger, GuiLogHandler
from mbo_utilities.gui.widgets.progress_bar import start_output_capture
from mbo_utilities.gui.widgets import get_supported_widgets, draw_all_widgets
from mbo_utilities import log

# Import modular components
from mbo_utilities.gui.widgets.menu_bar import draw_menu_bar, draw_keybinds_popup
from mbo_utilities.gui._popups import draw_tools_popups, draw_process_console_popup
from mbo_utilities.gui._save_as import draw_saveas_popup
from mbo_utilities.gui._keyboard import handle_keyboard_shortcuts, rebind_space_to_playback
from mbo_utilities.gui._dialogs import check_file_dialogs
from mbo_utilities.gui._stats import (
    compute_zstats,
    draw_stats_section,
    hydrate_zstats,
    refresh_zstats,
)
from mbo_utilities.gui._help_viewer import draw_help_popup
from mbo_utilities.gui._imgui_helpers import fit_width
from mbo_utilities.gui._metadata_editor import draw_metadata_popup
from mbo_utilities.gui._options_popup import draw_options_popup


import fastplotlib as fpl
from mbo_utilities.gui._edge_window import EdgeWindow
import contextlib

__all__ = ["PreviewDataWidget"]


import re as _re

_PLANE_DIR_RE = _re.compile(r"^plane\d+$", _re.IGNORECASE)


def _outdir_from_fpath(fpath) -> str | None:
    """Default output dir from a loaded fpath: the parent folder when fpath
    is a file, or the folder itself when fpath IS a directory. Used to
    seed the Run-tab output field so re-runs land alongside the source
    data unless the user explicitly browses elsewhere.
    """
    if fpath is None:
        return None
    if isinstance(fpath, (list, tuple)):
        if not fpath:
            return None
        fpath = fpath[0]
    try:
        p = Path(str(fpath))
    except (TypeError, ValueError):
        return None
    if not p.exists():
        return None
    return str(p if p.is_dir() else p.parent)


def _derive_suite2p_output_dir(fpath) -> str | None:
    """Detect a suite2p output location from a file or directory path.

    Returns the directory suite2p would have written into (the parent of
    the `plane*/` subdirs), or None if `fpath` doesn't look like a suite2p
    output. Used to auto-populate the GUI's output-folder field when the
    user opens an existing data.bin / ops.npy / volumetric results dir.

    Cases handled:
      - file inside `…/<root>/plane0/` (e.g. data.bin, ops.npy) → `<root>`
      - directory `…/<root>/plane0/`                            → `<root>`
      - directory `…/<root>/` containing one or more `plane*/`  → `<root>`
    """
    if fpath is None:
        return None
    if isinstance(fpath, (list, tuple)):
        if not fpath:
            return None
        fpath = fpath[0]
    try:
        p = Path(str(fpath))
    except (TypeError, ValueError):
        return None
    if not p.exists():
        return None

    parent = p.parent if p.is_file() else p
    if _PLANE_DIR_RE.match(parent.name):
        return str(parent.parent)
    try:
        for child in parent.iterdir():
            if child.is_dir() and _PLANE_DIR_RE.match(child.name):
                return str(parent)
    except (OSError, PermissionError):
        pass
    return None


def _base_5d(arr):
    """Peel the viewer's display wrappers (`_ScrubTimingProxy`,
    `_SqueezeSingletonDims`, a `FrameAveragedView`) back to the 5D array the
    frame-averaging view should wrap."""
    from mbo_utilities.arrays import FrameAveragedView
    from mbo_utilities.gui.run_gui import _ScrubTimingProxy, _SqueezeSingletonDims

    while True:
        if isinstance(arr, _ScrubTimingProxy):
            arr = arr._wrapped
        elif isinstance(arr, (_SqueezeSingletonDims, FrameAveragedView)):
            arr = arr._arr
        else:
            return arr


class PreviewDataWidget(EdgeWindow):
    """
    Main GUI widget for data preview and processing.

    This widget provides:
    - Time series data visualization with temporal/spatial projections
    - Z-stats signal quality analysis
    - Suite2p pipeline integration
    - File format conversion (tiff, zarr, etc.)

    Parameters
    ----------
    iw : fastplotlib.ImageWidget
        The ImageWidget to attach to.
    fpath : str | list | None
        Path(s) to the data file(s).
    threading_enabled : bool
        Whether to compute z-stats in background threads.
    size : int | None
        Width of the widget panel.
    location : str
        Panel location ("right" or "bottom").
    """

    # built on demand by sync_manual_roi() when the Widgets menu entry is on;
    # registers its panels on the top strip and owns the right-widget tabs
    manual_roi = None

    # the figure's top edge: menu row plus the registered full-width panels
    top_strip = None

    def __init__(
        self,
        iw: "fpl.ImageWidget",
        fpath: str | None | list = None,
        threading_enabled: bool = True,
        size: int | None = None,
        location: Literal["bottom", "right"] = "right",
        # None: no title bar over the panel. fastplotlib draws a custom,
        # full-width title box for an edge window with a title, and a static
        # label there only costs the panel a row of height.
        title: str | None = None,
        show_title: bool = False,
        movable: bool = False,
        resizable: bool = False,
        scrollable: bool = False,
        auto_resize: bool = True,
        window_flags: int | None = None,
        **kwargs,
    ):

        flags = (
            (imgui.WindowFlags_.no_title_bar if not show_title else 0)
            | (imgui.WindowFlags_.no_move if not movable else 0)
            | (imgui.WindowFlags_.no_resize if not resizable else 0)
            | (imgui.WindowFlags_.no_scrollbar if not scrollable else 0)
            | (imgui.WindowFlags_.no_scroll_with_mouse if not scrollable else 0)
            | (imgui.WindowFlags_.always_auto_resize if auto_resize else 0)
            | (window_flags or 0)
        )
        super().__init__(
            figure=iw.figure,
            size=250 if size is None else size,
            location=location,
            title=title,
            window_flags=flags,
        )

        # fastplotlib's ImguiFigure builds a second ImguiRenderer for its FPS
        # overlay, and ImguiRenderer.__init__ leaves *its* context current. Every
        # imgui/implot call below — implot context, style, fonts, io flags —
        # must target the context the figure actually renders with, or it lands
        # in an atlas nothing ever draws and pushing those fonts later kills the
        # render loop.
        imgui.set_current_context(iw.figure.imgui_renderer.imgui_context)

        # Initialize logging
        self._init_logging()

        # Initialize Suite2p settings
        self._init_suite2p()

        # Initialize ImPlot context
        if implot.get_current_context() is None:
            implot.create_context()

        # apply opaque imgui style (idempotent, runs once per process)
        from mbo_utilities.gui._imgui_helpers import style_imgui_opaque
        style_imgui_opaque()

        # Setup ImGui fonts
        self._init_fonts()

        # Store kwargs and paths
        self.kwargs = kwargs
        self.fpath = fpath if fpath else getattr(iw, "fpath", None)

        # Image widget setup
        self.image_widget = iw
        self.num_graphics = len(self.image_widget.graphics)
        self.shape = self.image_widget.data[0].shape

        # Determine data type (ScanImage or volumetric TIFF).
        # Peel the squeeze wrapper so isinstance sees the real class.
        from mbo_utilities.arrays import TiffArray
        first_arr = self.image_widget.data[0]
        underlying = getattr(first_arr, "_arr", first_arr)
        self.is_mbo_scan = (
            isinstance(underlying, ScanImageArray) or
            isinstance(underlying, TiffArray)
        )
        self.logger.debug(f"Data type: {type(first_arr).__name__}, is_mbo_scan: {self.is_mbo_scan}")

        # Initialize state
        self._init_state()

        # Initialize z-stats tracking
        self._init_zstats()

        # Initialize save dialog state
        self._init_saveas_state()

        # Initialize viewer
        self._init_viewer()

        # Hydrate cached stats from disk first; only compute the arrays that
        # had no valid cache (recompute on shape/dims/series mismatch).
        # compute_zstats runs its own single background worker, so call it
        # directly (no extra wrapper thread).
        if threading_enabled:
            hydrated = hydrate_zstats(self)
            pending = [i for i in range(self.num_graphics) if not hydrated[i]]
            if pending:
                self.logger.debug(f"Starting zstats computation for {len(pending)} array(s)...")
                for i in pending:
                    self._zstats_running[i] = True
                compute_zstats(self, only=pending)

    def _init_logging(self):
        """Initialize logging system."""
        self.debug_panel = GuiLogger()
        gui_handler = GuiLogHandler(self.debug_panel)
        gui_handler.setFormatter(logging.Formatter("%(message)s"))
        gui_handler.setLevel(logging.DEBUG)
        log.attach(gui_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        console_handler.setLevel(logging.DEBUG)

        if bool(int(os.getenv("MBO_DEBUG", "0"))):
            log.set_global_level(logging.DEBUG)
        else:
            log.set_global_level(logging.INFO)

        log.attach(console_handler)
        self.logger = log.get("gui")
        self.logger.debug("Logger initialized.")
        start_output_capture()

    def _init_suite2p(self):
        """Initialize Suite2p settings (lazy)."""
        # NOTE: removed the eager `start_preload()` call here. The preload
        # spawned a daemon thread that, despite a 1s sleep, ended up
        # contending for the GIL during fastplotlib's first-paint, so the
        # widget felt frozen for ~3.5s after the window appeared (suite2p
        # imports torch/numba/cellpose/scipy — ~3500 modules total).
        #
        # Pipeline registration is now fully lazy: `_register_pipelines()`
        # is called the first time the user opens the Pipeline Settings
        # popup, and it falls back to a synchronous import on that one
        # click. Trade-off: ~3.5s pause when the user FIRST opens the
        # popup, vs. instant widget startup. If you want the preload back
        # on a different trigger (e.g. when the user navigates to the Run
        # tab), call `start_preload()` from there.

        # defer dataclass creation until actually needed. all three are
        # lazy-initialized by matching properties below.
        self._s2p = None
        self._s2p_db = None
        self._s2p_extras = None
        self._s2p_savepath_flash_start = None
        self._s2p_savepath_flash_count = 0
        self._s2p_show_savepath_popup = False
        self._s2p_folder_dialog = None

    @property
    def s2p(self):
        """Suite2p processing settings (upstream schema)."""
        if self._s2p is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import Suite2pSettings
            from mbo_utilities.preferences import get_s2p_torch_device
            self._s2p = Suite2pSettings()
            # apply the persisted torch device as the sticky default (a loaded
            # dataset's settings.npy still overrides via _try_hydrate_s2p).
            self._s2p.torch_device = get_s2p_torch_device()
        return self._s2p

    @s2p.setter
    def s2p(self, value):
        self._s2p = value

    @property
    def s2p_db(self):
        """Suite2p input/output db (paths, plane counts) — upstream schema."""
        if self._s2p_db is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import Suite2pDB
            self._s2p_db = Suite2pDB()
        return self._s2p_db

    @s2p_db.setter
    def s2p_db(self, value):
        self._s2p_db = value

    @property
    def s2p_extras(self):
        """Mbo-only suite2p helper fields (dff_*, accept_all_cells, etc.)."""
        if self._s2p_extras is None and HAS_SUITE2P:
            from mbo_utilities.gui.widgets.pipelines.settings import MboSuite2pExtras
            self._s2p_extras = MboSuite2pExtras()
        return self._s2p_extras

    @s2p_extras.setter
    def s2p_extras(self, value):
        self._s2p_extras = value

    def _init_fonts(self):
        """Initialize ImGui fonts."""
        io = imgui.get_io()

        # disable keyboard nav so Space/Enter don't activate whatever button
        # currently holds nav focus (e.g. the EdgeWindow collapse caret)
        io.config_flags &= ~imgui.ConfigFlags_.nav_enable_keyboard

        fd_settings_dir = (
            Path(get_mbo_dirs()["imgui"])
            .joinpath("assets", "app_settings", "preview_settings.ini")
            .expanduser()
            .resolve()
        )
        io.set_ini_filename(str(fd_settings_dir))

        _fonts_dir = Path(imgui_bundle.__file__).parent.joinpath(
            "assets", "fonts", "Roboto"
        )
        sans_serif_font = str(_fonts_dir / "Roboto-Regular.ttf")
        self._default_imgui_font = io.fonts.add_font_from_file_ttf(
            sans_serif_font, 14, imgui.ImFontConfig()
        )
        # bold variant — used by the pipeline-settings popup to emphasize
        # "look at these first" parameters listed in _IMPORTANT_FIELDS,
        # paired with a thin box around the label. fall back to None if the
        # file is missing (the popup pushes the font only when non-None).
        bold_path = _fonts_dir / "Roboto-Bold.ttf"
        if bold_path.is_file():
            self._bold_font = io.fonts.add_font_from_file_ttf(
                str(bold_path), 14, imgui.ImFontConfig()
            )
        else:
            self._bold_font = None
        push_font_safe(self._default_imgui_font)

    def _init_state(self):
        """Initialize widget state."""
        for subplot in self.image_widget.figure:
            subplot.toolbar = False
        self.image_widget._sliders_ui._loop = True

        # Determine nz and nc (z-planes and channels) using dims property
        arr = self.image_widget.data[0]
        dims = getattr(arr, "dims", None)
        dims_lower = tuple(d.lower() for d in dims) if dims else None

        # detect z-planes
        if dims_lower is not None and "z" in dims_lower:
            z_idx = dims_lower.index("z")
            self.nz = self.shape[z_idx]
        elif dims_lower is not None and any(d in dims_lower for d in ("z-planes", "z-slices")):
            for d in ("z-planes", "z-slices"):
                if d in dims_lower:
                    z_idx = dims_lower.index(d)
                    self.nz = self.shape[z_idx]
                    break
        elif dims_lower is not None and "volumes" in dims_lower:
            # piezo: use num_slices if available, otherwise shape[1]
            if hasattr(arr, "num_slices"):
                self.nz = arr.num_slices
            else:
                vol_idx = dims_lower.index("volumes")
                if len(self.shape) >= 2 and self.shape[vol_idx] <= 1 and vol_idx + 1 < len(self.shape):
                    self.nz = self.shape[vol_idx + 1]
                else:
                    self.nz = self.shape[vol_idx]
        elif len(self.shape) >= 4 and (
            dims_lower is None
            or not any(d in dims_lower for d in ("channel", "channels", "c"))
        ):
            # only use shape[1] as z if not a channel dimension. bare "c" counts:
            # a squeezed TCZYX array with no Z reports ("T","C","Y","X"), and
            # reading shape[1] there returns the channel count as nz.
            self.nz = self.shape[1]
        else:
            self.nz = 1

        # detect channels (separate from z-planes)
        if dims_lower is not None and "channel" in dims_lower:
            c_idx = dims_lower.index("channel")
            self.nc = self.shape[c_idx]
        elif dims_lower is not None and "channels" in dims_lower:
            c_idx = dims_lower.index("channels")
            self.nc = self.shape[c_idx]
        elif dims_lower is not None and "c" in dims_lower:
            c_idx = dims_lower.index("c")
            self.nc = self.shape[c_idx]
        elif hasattr(arr, "num_color_channels"):
            self.nc = arr.num_color_channels
        else:
            self.nc = 1

        # detect camera views (isoview cm dimension)
        self.n_views = 1
        if dims_lower is not None and "cm" in dims_lower:
            cm_idx = dims_lower.index("cm")
            self.n_views = self.shape[cm_idx]

        self.logger.debug(f"Detected nz={self.nz}, nc={self.nc}, n_views={self.n_views} from dims={dims}")

        # Window/projection/contrast state — all per-data widget controls
        # are reset by _reset_per_data_state. Also called on every reload
        # in load_new_data, so initial-launch defaults and reload defaults
        # never drift.
        self._auto_update = False  # not data-specific, kept here
        from mbo_utilities.gui._dialogs import _reset_per_data_state
        _reset_per_data_state(self)

        # Registration state
        self._register_z = False
        self._register_z_progress = 0.0
        self._register_z_done = False
        self._register_z_running = False
        self._register_z_current_msg = ""
        # axial registration knobs; exposed in the save-as options popup.
        # default search radius matches compute_axial_shifts(max_reg_xy=30).
        self._axial_max_frames = 200
        self._axial_max_reg_xy = 30

        # Selection state
        self._selected_pipelines = None
        self._selected_array = 0
        self._selected_planes = None
        self._planes_str = ""

        # Settings menu flags
        self.show_debug_panel = False
        self.show_scope_window = False
        self.show_metadata_viewer = False
        self.show_diagnostics_window = False
        self._diagnostics_widget = None
        self._show_progress_overlay = True

        # Process monitoring
        self._viewing_process_pid = None

        # File dialogs
        self._file_dialog = None
        self._folder_dialog = None
        self._load_status_msg = ""
        self._load_status_color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)

        # Initialize Metadata Editor state
        self._custom_metadata = {}
        self._custom_key = ""
        self._custom_value = ""
        self._metadata_editor_open = False

        # Initialize widgets
        self._widgets = get_supported_widgets(self)

    def _init_zstats(self):
        """Initialize z-stats tracking state.

        Per-array slots are dicts keyed by the breakout-combo tuple (``()``
        when the array has no breakout dim); compute_zstats populates one
        entry per sampled (tile, camera, …) combination. ``_zstats_spec[i]``
        holds the `SummaryStatsSpec` used to map sliders back to a combo.
        """
        self._zstats = [{} for _ in range(self.num_graphics)]
        self._zstats_means = [{} for _ in range(self.num_graphics)]
        self._zstats_mean_scalar = [{} for _ in range(self.num_graphics)]
        self._zstats_spec = [None] * self.num_graphics
        self._zstats_done = [False] * self.num_graphics
        self._zstats_running = [False] * self.num_graphics
        self._zstats_progress = [0.0] * self.num_graphics
        self._zstats_current_z = [0] * self.num_graphics
        self._zstats_z_indices = [None] * self.num_graphics
        # series axis when both zplanes and timepoints exist ("z" or "t")
        self._stats_axis_pref = "z"

    def _init_saveas_state(self):
        """Initialize save-as dialog state."""
        self._ext = ".tiff"
        self._ext_idx = MBO_AVAILABLE_FTYPES.index(".tiff")
        self._overwrite = True
        self._debug = False
        self._saveas_chunk_mb = 100

        # Zarr options
        self._zarr_sharded = True
        self._zarr_ome = True
        self._zarr_compression_level = 1
        self._zarr_pyramid = False
        self._zarr_pyramid_max_layers = 4
        self._zarr_pyramid_method = "median"

        # H5 options
        # dataset name inside the .h5 file. suite2p reads from "mov" by
        # default and lbm_suite2p_python expects the same; only change this
        # if you're targeting a downstream tool that wants something else.
        self._h5_dataset_name = "mov"

        # Save dialog state
        self._saveas_popup_open = False
        self._saveas_done = False
        self._saveas_running = False
        self._saveas_progress = 0.0
        self._saveas_current_index = 0

        # Set Metadata popup — driven by File menu and Shift+M shortcut
        # (see gui/_metadata_editor.draw_metadata_popup).
        self._show_metadata_popup = False
        # Widgets menu: the manual ROI widget while it is on (see
        # sync_manual_roi / gui/manual_roi.attach_roi_widget)
        self.manual_roi = None

        # Directories
        save_as_dir = get_last_dir("save_as")
        self._saveas_outdir = str(save_as_dir) if save_as_dir else ""

        # Default suite2p output dir = the loaded path's parent (file) or
        # the folder itself (directory). Most intuitive default — re-runs
        # land next to the source data. Falls back to the cached last-used
        # dir only when nothing is loaded yet.
        _loaded_outdir = _outdir_from_fpath(self.fpath)
        if _loaded_outdir:
            self._s2p_outdir = _loaded_outdir
        else:
            s2p_output_dir = get_last_dir("suite2p_output")
            self._s2p_outdir = str(s2p_output_dir) if s2p_output_dir else ""

        # If the loaded data lives inside a suite2p output tree (data.bin
        # in plane*/, ops.npy / stat.npy / etc.), or is a volumetric root
        # containing plane*/, point _s2p_outdir at that root so a re-run
        # lands in the same place. Wins over the parent-folder default.
        _derived_s2p_dir = _derive_suite2p_output_dir(self.fpath)
        if _derived_s2p_dir:
            self._s2p_outdir = _derived_s2p_dir

        # Hydrate Suite2pSettings/DB/Extras from a sibling settings.npy /
        # db.npy / ops.npy when the loaded data is a suite2p output. The
        # file-menu Open path does this in load_new_data; doing it here
        # covers the CLI launch (`mbo path/to/data.bin` → fpath comes from
        # data_array.source_path which is the plane / volume directory).
        try:
            from mbo_utilities.gui._dialogs import _try_hydrate_s2p_from_binary
            _try_hydrate_s2p_from_binary(self, self.fpath)
        except Exception as _e:
            self.logger.debug(f"suite2p hydrate (init): {_e}")

        self._saveas_folder_dialog = None
        self._saveas_total = 0

        # ROI selection
        self._saveas_selected_roi = set()
        self._saveas_rois = False
        self._saveas_selected_roi_mode = "All"

        # Metadata
        self._saveas_custom_metadata = {}
        self._saveas_custom_key = ""
        self._saveas_custom_value = ""

        # Output suffix
        self._saveas_output_suffix = ""

        # Options
        self._saveas_background = True

        # Scan-phase correction for save/export (separate from display settings)
        # defaults to True for save operations
        self._saveas_fix_phase = True
        self._saveas_use_fft = True

        # Video export options (active when ext is .mp4)
        self._saveas_video_fps = 30
        self._saveas_video_speed_factor = 1.0
        self._saveas_video_auto = True
        self._saveas_video_vmin = 0.0
        self._saveas_video_vmax = 1000.0
        self._saveas_video_vmin_pct = 1.0
        self._saveas_video_vmax_pct = 99.5
        self._saveas_video_temporal_smooth = 0
        self._saveas_video_temporal_mode_idx = 0  # 0=mean 1=max 2=std
        self._saveas_video_spatial_smooth = 0.0
        self._saveas_video_gamma = 1.0
        from mbo_utilities.gui._colormaps import DEFAULT_COLORMAPS, DEFAULT_COLORMAP
        self._saveas_video_cmaps: list[str] = list(DEFAULT_COLORMAPS)
        self._saveas_video_cmap_idx: int = self._saveas_video_cmaps.index(DEFAULT_COLORMAP)
        self._saveas_video_quality_idx = 2  # 0=preview 1=high 2=visually lossless 3=lossless
        self._saveas_video_codec_idx = 0  # 0 = libx264
        self._saveas_video_mean_subtract = False
        self._saveas_video_time_overlay = False
        self._saveas_video_scalebar = False
        self._saveas_video_upscale = 0  # 0 = auto

    def _init_top_strip(self):
        """Claim the figure's top edge for the menu row.

        The strip spans the canvas's full width, so it is also where the
        panels that want that width register themselves — Manual ROI's ROI
        and Traces cards, the Signal Quality plot.
        """
        from mbo_utilities.gui._top_strip import TopStrip

        self.top_strip = TopStrip(
            self.image_widget.figure, draw_menu=lambda: draw_menu_bar(self)
        )

    def _sync_top_panels(self) -> None:
        """Register / drop the top panels this widget owns, per frame."""
        from mbo_utilities.gui._top_strip import TopPanel
        from mbo_utilities.gui.widgets.widget_toggles import widget_enabled

        want = widget_enabled("signal_quality") and any(self._zstats_done)
        if want and not self.top_strip.has("zstats"):
            self.top_strip.register(
                TopPanel(
                    "zstats", "Signal Quality", self.draw_stats_plot,
                    height=260, right_tab="signal_quality", priority=20,
                )
            )
        elif not want:
            self.top_strip.unregister("zstats")

    def _init_viewer(self):
        """Initialize the viewer based on data type."""
        from mbo_utilities.gui.viewers import get_viewer_class
        self._init_top_strip()
        viewer_cls = get_viewer_class(self.image_widget.data[0])
        self._viewer = viewer_cls(self.image_widget, self.fpath, parent=self)
        self.logger.debug(f"Viewer: {self._viewer.name}")
        self.set_context_info()
        self._update_window_funcs()
        rebind_space_to_playback(self)
        self._seed_playback_fps()
        try:
            from mbo_utilities.gui.widgets.pipelines.isoview import (
                maybe_spawn_raw_projections,
            )
            maybe_spawn_raw_projections(self)
        except Exception:
            self.logger.debug("raw projection prefetch skipped", exc_info=True)
        # honour a persisted / CLI-set "Manual ROI Labeling" toggle
        from mbo_utilities.gui.widgets.widget_toggles import widget_enabled
        self.sync_manual_roi(widget_enabled("manual_roi"))

    def sync_manual_roi(self, enabled: bool) -> None:
        """Create or tear down the manual-ROI widget to match the toggle.

        Building it attaches an overlay graphic to the subplot and claims the
        figure's top strip and right-widget tabs, so it is created lazily the
        first time the widget is switched on and dropped again when it is
        switched off.
        """
        from mbo_utilities.gui.manual_roi import attach_roi_widget, detach_roi_widget

        current = getattr(self, "manual_roi", None)
        if enabled and current is None:
            # attach logs (never raises) when the widget cannot be built, and
            # adopts the store parked by the previous detach in this session
            attach_roi_widget(self)
        elif not enabled and current is not None:
            try:
                detach_roi_widget(self)
            except Exception:
                self.logger.debug("manual ROI teardown failed", exc_info=True)
                self.manual_roi = None

    def _seed_playback_fps(self):
        """Seed the t playback rate from the loaded array's frame rate.

        Uses the vendored slider widget's `seed_fps`, which clamps to the
        slider's [1, 50] range and never overrides a user-typed fps.
        """
        try:
            arr = self.image_widget.data[0]
            fs = getattr(arr, "fs", None)
            if fs is None:
                from mbo_utilities.metadata import get_param
                fs = get_param(getattr(arr, "metadata", None), "fs")
            from mbo_utilities.gui._keyboard import _get_sliders_ui
            sliders = _get_sliders_ui(self)
            if sliders is not None and hasattr(sliders, "seed_fps"):
                sliders.seed_fps("t", fs)
        except Exception:
            self.logger.debug("playback fps seed skipped", exc_info=True)

    def set_context_info(self):
        """Update app title with dataset name."""
        try:
            if self.fpath is None:
                return
            from mbo_utilities import __version__
            name = Path(self.fpath[0]).parent.name if isinstance(self.fpath, list) else Path(self.fpath).name
            hello_imgui.get_runner_params().app_shallow_settings.window_title = f"Miller Brain Studio v{__version__} - {name}"
        except (RuntimeError, TypeError):
            pass

    # === Properties ===

    @property
    def processors(self) -> list:
        """Access to underlying NDImageProcessor instances.

        Empty on stock fastplotlib (no processors API); every consumer
        early-returns on an empty list, degrading projection/window
        controls instead of crashing.
        """
        return getattr(self.image_widget, "_image_processors", [])

    def _get_data_arrays(self) -> list:
        """Get underlying data arrays from image processors."""
        procs = self.processors
        if procs:
            return [proc.data for proc in procs]
        return list(getattr(self.image_widget, "data", None) or [])

    @property
    def current_offset(self) -> list[float]:
        """Scan-phase offset of the *currently displayed* frame, per graphic.

        Reads from each array's per-(t, c, z) offset cache via
        `arr.get_offset_at(t, c, z)` rather than `arr.offset`. The latter
        returns `_last_offset`, which is global mutable state clobbered by
        every read on the array — including background reads from histogram
        subsamplers and zstats workers — so it drifts even when the user
        isn't scrubbing. The cached lookup, by contrast, is keyed to the
        exact (t, c, z) cell on screen, so the displayed value only changes
        when the user actually moves a slider.
        """
        # current displayed indices, defaulting to 0 when a slider is absent
        # (e.g. T-only data has no z slider, so we report z=0).
        from mbo_utilities.arrays.features import find_slider_name
        indices = {}
        names = ()
        if self.image_widget is not None:
            try:
                names = self.image_widget._slider_dim_names or ()
                for name in names:
                    try:
                        indices[name] = int(self.image_widget.indices[name])
                    except (KeyError, IndexError, TypeError):
                        pass
            except Exception:
                pass
        t_name = find_slider_name(names, "t")
        c_name = find_slider_name(names, "c")
        z_name = find_slider_name(names, "z")
        t_idx = indices.get(t_name, 0) if t_name else 0
        c_idx = indices.get(c_name, 0) if c_name else 0
        z_idx = indices.get(z_name, 0) if z_name else 0

        offsets = []
        for arr in self._get_data_arrays():
            value = None
            getter = getattr(arr, "get_offset_at", None)
            if callable(getter):
                value = getter(t_idx, c_idx, z_idx)
            if value is None:
                # cache miss (frame hasn't been read yet under current
                # settings) or array doesn't expose the per-frame cache —
                # leave the display at 0.0 rather than showing stale state.
                offsets.append(0.0)
            else:
                offsets.append(float(value))
        return offsets

    @property
    def has_raster_scan_support(self) -> bool:
        """Check if any data array supports raster scan phase correction."""
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                return True
            if hasattr(arr, "fix_phase") and hasattr(arr, "use_fft"):
                return True
        return False

    @property
    def fix_phase(self) -> bool:
        """Whether bidirectional phase correction is enabled."""
        arrays = self._get_data_arrays()
        if not arrays:
            return False
        arr = arrays[0]
        if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.enabled
        return getattr(arr, "fix_phase", False)

    @fix_phase.setter
    def fix_phase(self, value: bool):
        self.logger.debug(f"Setting fix_phase to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.enabled = value
            elif hasattr(arr, "fix_phase"):
                arr.fix_phase = value
        self._refresh_image_widget()

    @property
    def use_fft(self) -> bool:
        """Whether FFT-based phase correlation is used."""
        arrays = self._get_data_arrays()
        if not arrays:
            return False
        arr = arrays[0]
        if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.use_fft
        return getattr(arr, "use_fft", False)

    @use_fft.setter
    def use_fft(self, value: bool):
        self.logger.debug(f"Setting use_fft to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.use_fft = value
            elif hasattr(arr, "use_fft"):
                arr.use_fft = value
        self._refresh_image_widget()

    @property
    def border(self) -> int:
        """Border pixels to exclude from phase correlation."""
        arrays = self._get_data_arrays()
        if not arrays:
            return 3
        arr = arrays[0]
        if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
            return arr.phase_correction.border
        return getattr(arr, "border", 3)

    @border.setter
    def border(self, value: int):
        self.logger.debug(f"Setting border to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "phase_correction") and isinstance(arr.phase_correction, PhaseCorrectionFeature):
                arr.phase_correction.border = value
            elif hasattr(arr, "border"):
                arr.border = value
        self._refresh_image_widget()

    @property
    def max_offset(self) -> int:
        """Maximum pixel offset for phase correction."""
        arrays = self._get_data_arrays()
        return getattr(arrays[0], "max_offset", 3) if arrays else 3

    @max_offset.setter
    def max_offset(self, value: int):
        self.logger.debug(f"Setting max_offset to {value}.")
        for arr in self._get_data_arrays():
            if hasattr(arr, "max_offset"):
                arr.max_offset = value
        self._refresh_image_widget()

    @property
    def gaussian_sigma(self) -> float:
        """Sigma for Gaussian blur (0 = disabled)."""
        return self._gaussian_sigma

    @gaussian_sigma.setter
    def gaussian_sigma(self, value: float):
        self._gaussian_sigma = max(0.0, value)
        self._rebuild_spatial_func()
        self._refresh_image_widget()

    @property
    def proj(self) -> str:
        """Current projection mode (mean, max, std)."""
        return self._proj

    @proj.setter
    def proj(self, value: str):
        if value != self._proj:
            self.logger.debug(f"Projection changed: {self._proj} -> {value}")
            self._proj = value
            self._update_window_funcs()

    @property
    def mean_subtraction(self) -> bool:
        """Whether mean subtraction is enabled."""
        return self._mean_subtraction

    @mean_subtraction.setter
    def mean_subtraction(self, value: bool):
        if value != self._mean_subtraction:
            self._mean_subtraction = value
            self._update_mean_subtraction()

    @property
    def auto_contrast_on_z(self) -> bool:
        """Whether to auto-reset contrast when z-plane changes."""
        return self._auto_contrast_on_z

    @auto_contrast_on_z.setter
    def auto_contrast_on_z(self, value: bool):
        self._auto_contrast_on_z = value

    @property
    def frame_average(self) -> int:
        """Source frames averaged into each displayed frame (1 = off).

        Unlike the window functions, which project over a sliding window for
        display only, this is a step in the data pipeline: the array itself
        is wrapped so every consumer — the display, the ROI traces, extract /
        demix, the pipelines and the writers — sees the averaged frames, the
        way ``fix_phase`` is baked in before anything downstream reads.
        """
        return self._frame_average

    @frame_average.setter
    def frame_average(self, value: int):
        value = max(1, int(value))
        source = self._frame_average_source or _base_5d(self.image_widget.data[0])
        nt = int(source.shape[0]) if len(source.shape) == 5 else 1
        value = min(value, max(nt, 1))
        if value == self._frame_average:
            return
        self._apply_frame_average(source, value)

    def _apply_frame_average(self, source, factor: int) -> None:
        """Swap the viewer onto (or off) a ``FrameAveragedView`` of ``source``.

        T changes, so this is the same swap File -> Open does — stale window
        funcs and spatial closures dropped, indices reset, z-stats recomputed,
        playback re-seeded — with one difference: the ROI store survives,
        because masks are (Z, Y, X) and averaging only touches T.
        """
        from mbo_utilities.arrays import average_frames
        from mbo_utilities.gui.manual_roi import attach_roi_widget, detach_roi_widget
        from mbo_utilities.gui.run_gui import _squeeze_for_viewer

        iw = self.image_widget
        wrapped = _squeeze_for_viewer(average_frames(source, factor))

        roi_was_on = getattr(self, "manual_roi", None) is not None
        if roi_was_on:
            try:
                detach_roi_widget(self)  # keeps the store and parked runs
            except Exception:
                self.logger.debug("manual ROI teardown failed", exc_info=True)
                self.manual_roi = None

        # the window funcs and the spatial closure are bound to the old t-rank
        self._window_size = 1
        if hasattr(self, "_rebuild_spatial_func"):
            self._rebuild_spatial_func()
        for proc in getattr(iw, "_image_processors", []):
            proc.window_funcs = None
            proc.window_sizes = None
            proc.window_order = None

        iw.data[0] = wrapped
        if iw.n_sliders > 0:
            iw.indices = [0] * iw.n_sliders

        self._frame_average = factor
        self._frame_average_source = source if factor > 1 else None
        self.shape = wrapped.shape
        if len(self.shape) == 5:
            self.nc, self.nz = self.shape[1], self.shape[2]
        elif len(self.shape) == 4:
            self.nc, self.nz = 1, self.shape[1]
        else:
            self.nc = self.nz = 1
        self.set_context_info()
        self._refresh_widgets()

        try:
            self._seed_playback_fps()
        except Exception:
            self.logger.debug("playback fps seed skipped", exc_info=True)
        try:
            self.refresh_zstats()
        except Exception:
            self.logger.debug("zstats refresh skipped", exc_info=True)

        if roi_was_on:
            attach_roi_widget(self)
        self.logger.info(
            f"frame averaging {'off' if factor == 1 else factor}: shape {self.shape}"
        )

    @property
    def window_size(self) -> int:
        """Window size for temporal projection."""
        return self._window_size

    @window_size.setter
    def window_size(self, value: int):
        # clamp to available timepoints
        nt = self.shape[0] if len(self.shape) > 0 else 1
        value = max(1, min(value, nt))
        if value == self._window_size:
            return
        self._window_size = value
        self.logger.debug(f"Window size set to {value}.")
        if not self.processors:
            self._apply_legacy_window_funcs()
            return
        n_slider_dims = self.processors[0].n_slider_dims
        if n_slider_dims == 0:
            return
        per_processor_sizes = (self._window_size,) + (None,) * (n_slider_dims - 1)
        self._set_processor_attr("window_sizes", per_processor_sizes)

    # === Internal methods ===

    def _refresh_image_widget(self):
        """Trigger a frame refresh on the ImageWidget."""
        current_indices = list(self.image_widget.indices)
        self.image_widget.indices = current_indices

    def _set_processor_attr(self, attr: str, value):
        """Set processor attribute without expensive histogram recomputation."""
        if not self.processors:
            self._set_legacy_iw_attr(attr, value)
            return

        # save histogram state and disable recomputation during update
        saved = [(p._compute_histogram, p._histogram) for p in self.processors]
        for proc in self.processors:
            proc._compute_histogram = False

        try:
            if attr == "window_funcs":
                if isinstance(value, tuple):
                    value = [value] * len(self.processors)
                self.image_widget.window_funcs = value
            elif attr == "window_sizes":
                if isinstance(value, (tuple, list)) and not isinstance(value[0], (tuple, list, type(None))):
                    value = [value] * len(self.processors)
                self.image_widget.window_sizes = value
            elif attr == "spatial_func":
                self.image_widget.spatial_func = value
            else:
                for proc in self.processors:
                    setattr(proc, attr, value)
                self._refresh_image_widget()
        except Exception as e:
            self.logger.exception(f"Error setting {attr}: {e}")
        finally:
            # restore histogram state
            for proc, (orig_compute, orig_hist) in zip(self.processors, saved):
                proc._compute_histogram = orig_compute
                proc._histogram = orig_hist
            # fix dock visibility (fastplotlib sets to 0 when histogram is None)
            for subplot in self.image_widget.figure:
                if subplot.docks["right"].size < 1:
                    subplot.docks["right"].size = 80

        # fpl's window_funcs/window_sizes/spatial_func setters refresh via
        # `self.indices = self.indices`, but that runs *before* the histogram
        # restore above and can leave the displayed graphic stale. Force a
        # post-restore refresh so changes take effect immediately.
        if attr in ("window_funcs", "window_sizes", "spatial_func"):
            self._refresh_image_widget()

    def _set_legacy_iw_attr(self, attr: str, value):
        """No processors API (stock ImageWidget): translate to its native
        ``window_funcs`` / ``frame_apply``."""
        iw = self.image_widget
        if not hasattr(iw, "window_funcs"):
            return
        try:
            if attr in ("window_funcs", "window_sizes"):
                self._apply_legacy_window_funcs()
            elif attr == "spatial_func":
                if isinstance(value, (list, tuple)):
                    funcs = list(value)
                else:
                    funcs = [value] * self.num_graphics
                iw.frame_apply = {
                    i: f for i, f in enumerate(funcs) if f is not None
                }
        except Exception as e:
            self.logger.exception(f"Error setting {attr}: {e}")

    def _apply_legacy_window_funcs(self):
        """Map projection mode + window size onto the stock ImageWidget's
        ``window_funcs`` ({"t": (func, size)}). Its half-window math yields
        an empty window for even/size<3 values, so sizes are odd-ified and
        size<=1 clears the projection (raw frame)."""
        iw = self.image_widget
        if not hasattr(iw, "window_funcs"):
            return
        if "t" not in (getattr(iw, "slider_dims", None) or ()):
            return
        try:
            size = int(self._window_size)
            if size <= 1:
                iw.window_funcs = None
                return
            proj_funcs = {"mean": np.mean, "max": np.max, "std": np.std}
            func = proj_funcs.get(self._proj, np.mean)
            iw.window_funcs = {"t": (func, max(3, size | 1))}
        except Exception as e:
            self.logger.exception(f"Error applying window funcs: {e}")

    def _refresh_widgets(self):
        """Refresh widgets based on current data capabilities."""
        self._widgets = get_supported_widgets(self)

    def _update_mean_subtraction(self):
        """Update spatial_func to apply mean subtraction.

        does NOT reset vmin/vmax — callers that need a contrast reset
        (e.g. the mean_subtraction setter on initial toggle) must do so
        explicitly. this keeps per-z-plane rebuilds from silently
        overriding the user's auto_contrast_on_z preference.
        """
        self._rebuild_spatial_func()
        self._refresh_image_widget()

    def _rebuild_spatial_func(self):
        """Rebuild and apply the combined spatial function."""
        from mbo_utilities.arrays.features import find_slider_name
        names = self.image_widget._slider_dim_names or ()
        # fastplotlib's `indices` is case-sensitive; isoview uses
        # descriptive labels (Tile/Timepoint, Cam/View, Zplane), LBM
        # uses lowercase. find_slider_name resolves both via aliases.
        z_name = find_slider_name(names, "z")
        try:
            z_idx = self.image_widget.indices[z_name] if z_name else 0
        except (IndexError, KeyError):
            z_idx = 0

        sigma = self.gaussian_sigma if self.gaussian_sigma > 0 else None

        # Pick the breakout combo matching the current sliders so mean
        # subtraction matches what the user sees. `_zstats_means[i]` is now
        # `dict[combo_tuple, (n_slices, Yb, Xb)]`; fall back to the
        # no-breakout slot, then the first computed combo.
        from mbo_utilities.gui._stats import current_breakout_key

        def _means_for(i):
            slot = self._zstats_means[i] if i < len(self._zstats_means) else None
            if not isinstance(slot, dict) or not slot:
                return None
            key = current_breakout_key(self, i)
            out = slot.get(key)
            if out is None:
                out = slot.get(())
            if out is None:
                out = next(iter(slot.values()))
            return out

        any_mean_sub = self._mean_subtraction and any(
            self._zstats_done[i] and _means_for(i) is not None
            for i in range(self.num_graphics)
        )

        if not any_mean_sub and sigma is None:
            def identity(frame):
                return frame
            self._set_processor_attr("spatial_func", identity)
            return

        spatial_funcs = []
        for i in range(self.num_graphics):
            mean_img = None
            if self._mean_subtraction and self._zstats_done[i]:
                means_arr = _means_for(i)
                if means_arr is not None:
                    # means_arr rows follow the sampled planes, which may be
                    # a strided subset for deep stacks. map the displayed
                    # plane to its nearest sampled row instead of indexing by
                    # the absolute plane (which would be out of range).
                    pos = self._sampled_mean_pos(i, z_idx, means_arr.shape[0])
                    mean_img = means_arr[pos].astype(np.float32)

            spatial_funcs.append(self._make_spatial_func(mean_img, sigma))

        self._set_processor_attr("spatial_func", spatial_funcs)

    def _sampled_mean_pos(self, i: int, z_idx: int, n_rows: int) -> int:
        """Row in graphic ``i``'s mean-images stack for displayed plane z_idx.

        When z is subsampled the stack has fewer rows than the volume has
        planes; map the absolute plane to the nearest sampled plane. Falls
        back to a clamped index when no sampling record is available.
        """
        idxs = getattr(self, "_zstats_z_indices", None)
        if idxs and i < len(idxs) and idxs[i] and len(idxs[i]) == n_rows:
            target = int(z_idx) + 1  # records are 1-based plane numbers
            planes = idxs[i]
            return min(range(n_rows), key=lambda k: abs(planes[k] - target))
        return max(0, min(int(z_idx), n_rows - 1))

    def _make_spatial_func(self, mean_img: np.ndarray | None, sigma: float | None):
        """Create a spatial function that applies mean subtraction and/or gaussian blur."""
        # precompute kernel size for opencv (6*sigma, rounded to odd)
        ksize = (int(sigma * 6) | 1) if sigma else 0
        # zstats mean-images are spatially binned (strided) to save memory;
        # block-upsample back to the frame grid before subtracting. cached
        # per frame shape so it is computed once, not every frame.
        _fitted: dict = {}

        def _fit_mean(shape):
            if mean_img.shape == shape:
                return mean_img
            cached = _fitted.get(shape)
            if cached is None:
                fy = -(-shape[0] // mean_img.shape[0])
                fx = -(-shape[1] // mean_img.shape[1])
                up = np.repeat(np.repeat(mean_img, fy, axis=0), fx, axis=1)
                cached = np.ascontiguousarray(up[: shape[0], : shape[1]])
                _fitted[shape] = cached
            return cached

        def spatial_func(frame):
            # fastplotlib passes the raw data object when n_slider_dims==0,
            # which for our lazy arrays is not yet a numpy array.
            # materialize first so arithmetic/ufuncs work.
            result = np.asarray(frame)
            if mean_img is not None and result.ndim == 2:
                # only subtract when frame is 2D (Y, X); skip if 3D since
                # mean_img is z-specific and can't be applied to a full stack
                result = result.astype(np.float32) - _fit_mean(result.shape)
            if sigma is not None and sigma > 0 and result.ndim == 2:
                try:
                    import cv2
                    result = cv2.GaussianBlur(result, (ksize, ksize), sigma)
                except ImportError:
                    result = gaussian_filter(result, sigma=sigma)
            return result
        return spatial_func

    def _update_window_funcs(self):
        """Update window_funcs on image widget based on current projection mode."""
        if not self.processors:
            self._apply_legacy_window_funcs()
            return

        def mean_wrapper(data, axis, keepdims):
            return np.mean(data, axis=axis, keepdims=keepdims)

        def max_wrapper(data, axis, keepdims):
            return np.max(data, axis=axis, keepdims=keepdims)

        def std_wrapper(data, axis, keepdims):
            return np.std(data, axis=axis, keepdims=keepdims)

        proj_funcs = {"mean": mean_wrapper, "max": max_wrapper, "std": std_wrapper}
        proj_func = proj_funcs.get(self._proj, mean_wrapper)

        n_slider_dims = self.processors[0].n_slider_dims

        if n_slider_dims == 0:
            return
        elif n_slider_dims == 1:
            window_funcs = (proj_func,)
        elif n_slider_dims == 2:
            window_funcs = (proj_func, None)
        else:
            window_funcs = (proj_func,) + (None,) * (n_slider_dims - 1)

        self._set_processor_attr("window_funcs", window_funcs)

    def gui_progress_callback(self, frac, meta=None):
        """Handle progress callbacks from save operations."""
        if isinstance(meta, (int, np.integer)):
            self._saveas_progress = frac
            self._saveas_current_index = meta
            self._saveas_done = frac >= 1.0
            if frac >= 1.0:
                self._saveas_running = False
                self._saveas_complete_time = time.time()
                self.logger.info("Save complete")
        elif isinstance(meta, str):
            self._register_z_progress = frac
            self._register_z_current_msg = meta
            self._register_z_done = frac >= 1.0
            if frac >= 1.0:
                self._register_z_running = False
                self._register_z_complete_time = time.time()

    def _clear_stale_progress(self):
        """Clear completed progress indicators after a delay."""
        now = time.time()
        clear_delay = 5.0

        if getattr(self, "_saveas_done", False):
            complete_time = getattr(self, "_saveas_complete_time", 0)
            if now - complete_time > clear_delay:
                self._saveas_done = False
                self._saveas_progress = 0.0

        if getattr(self, "_register_z_done", False):
            complete_time = getattr(self, "_register_z_complete_time", 0)
            if now - complete_time > clear_delay:
                self._register_z_done = False
                self._register_z_progress = 0.0
                self._register_z_current_msg = None

    # === Rendering ===

    def draw_window(self):
        """Override parent to handle keyboard shortcuts and global popups."""
        handle_keyboard_shortcuts(self)
        check_file_dialogs(self)

        # Draw independent floating windows
        draw_tools_popups(self)
        draw_saveas_popup(self)
        draw_metadata_popup(self)
        draw_process_console_popup(self)
        draw_keybinds_popup(self)
        draw_help_popup(self)
        draw_options_popup(self)
        from mbo_utilities.gui._cloud import draw_cloud_popup
        from mbo_utilities.gui.widgets.biohpc import draw_biohpc_popup

        draw_biohpc_popup(self)
        draw_cloud_popup(self)
        try:
            from mbo_utilities.gui.widgets.isoview_crop import draw_window as _draw_iso_crop_window
            _draw_iso_crop_window(self)
        except Exception:
            # Optional widget — skip silently when its deps aren't loaded
            # (e.g. immvision unavailable in the running imgui_bundle).
            pass
        try:
            from mbo_utilities.gui.widgets.isoview_segment import draw_window as _draw_iso_seg_window
            _draw_iso_seg_window(self)
        except Exception:
            pass
        try:
            from mbo_utilities.gui.widgets.isoview_deadpixel import draw_window as _draw_iso_dp_window
            _draw_iso_dp_window(self)
        except Exception:
            pass

        super().draw_window()

    def update(self):
        """Main render callback."""
        import time
        t0 = time.perf_counter()
        # `gap` measures wall-clock time since the previous frame entered
        # update(). On a healthy GUI this should hover near the canvas's
        # frame interval (16 ms @ 60 Hz). A spike means the main thread
        # was blocked between frames — by GIL contention with a worker,
        # the canvas swap, a Qt event handler, etc. Logging this is the
        # primary signal for "GUI feels frozen during zstats."
        prev = getattr(self, "_last_frame_t", None)
        gap_ms = (t0 - prev) * 1000.0 if prev is not None else 0.0
        self._last_frame_t = t0

        # Surface the Projections widget once the background raw-projection
        # worker has written its output (widget list is otherwise only built
        # on load). Cheap: gated to one refresh per raw dir.
        try:
            from mbo_utilities.gui.widgets.pipelines.isoview import (
                maybe_refresh_raw_projections,
            )
            maybe_refresh_raw_projections(self)
        except Exception:
            self.logger.debug("raw projection refresh skipped", exc_info=True)

        self._sync_top_panels()
        t1 = time.perf_counter()
        # wrap text at the panel edge everywhere the viewer draws directly;
        # each tab's child re-applies it (imgui wraps per window)
        with fit_width():
            self._viewer.draw()
        t2 = time.perf_counter()
        menu_ms = (t1 - t0) * 1000  # top-panel bookkeeping, the menu draws in the strip
        draw_ms = (t2 - t1) * 1000
        if gap_ms > 100 or menu_ms > 50 or draw_ms > 50:
            self.logger.debug(
                f"SLOW FRAME: gap={gap_ms:.0f}ms menu={menu_ms:.1f}ms draw={draw_ms:.1f}ms"
            )

        # Update mean subtraction when z-plane or channel changes. With
        # all channels precomputed by zstats, a C change just rebinds the
        # mean image from the per-channel cache — no recompute needed.
        # `indices` is case-sensitive; resolve canonical names from the
        # slider list before indexing (isoview uses descriptive labels,
        # LBM uses lowercase t/c/z).
        from mbo_utilities.arrays.features import find_slider_name
        names = self.image_widget._slider_dim_names or ()
        z_name = find_slider_name(names, "z")
        c_name = find_slider_name(names, "c")
        try:
            z_idx = self.image_widget.indices[z_name] if z_name else 0
        except (IndexError, KeyError):
            z_idx = 0
        try:
            c_idx = self.image_widget.indices[c_name] if c_name else 0
        except (IndexError, KeyError):
            c_idx = 0

        if z_idx != self._last_z_idx or c_idx != getattr(self, "_last_c_idx", 0):
            self._last_z_idx = z_idx
            self._last_c_idx = c_idx
            # rebuild mean-sub (if active) and reset contrast (if auto) are
            # independent concerns — both can apply on the same z change.
            if self._mean_subtraction:
                self._update_mean_subtraction()
            if self._auto_contrast_on_z and self.image_widget:
                self.image_widget.reset_vmin_vmax_frame()

    def draw_stats_section(self):
        """The Signal Quality tab: the metric table (the plot is the top
        strip's ``Signal Quality`` panel, which has the width for it)."""
        draw_stats_section(self, plot=False)

    def draw_stats_plot(self):
        """The Signal Quality top panel: the plot, full canvas width."""
        draw_stats_section(self, table=False)

    def draw_preview_section(self):
        """Draw preview section using modular UI widgets."""
        imgui.dummy(imgui.ImVec2(0, 5))
        with imgui_ctx.begin_child("##PreviewChild", imgui.ImVec2(0, 0), imgui.ChildFlags_.none):
            with fit_width():
                draw_all_widgets(self, self._widgets)

    def compute_zstats(self):
        """Compute z-stats for all graphics."""
        compute_zstats(self)

    def refresh_zstats(self):
        """Reset and recompute z-stats for all arrays."""
        refresh_zstats(self)

    def cleanup(self):
        """Clean up resources when the GUI is closing."""
        from mbo_utilities.gui.widgets.pipelines import cleanup_pipelines
        from mbo_utilities.gui.widgets import cleanup_all_widgets

        cleanup_pipelines(self)
        cleanup_all_widgets(self._widgets)
        self.sync_manual_roi(False)
        strip = getattr(self, "top_strip", None)
        if strip is not None:
            strip.close()
            self.top_strip = None

        self._file_dialog = None
        self._folder_dialog = None
        if hasattr(self, "_s2p_folder_dialog"):
            self._s2p_folder_dialog = None

        self.logger.debug("GUI cleanup complete")
