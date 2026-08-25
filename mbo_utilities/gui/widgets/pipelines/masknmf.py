"""masknmf pipeline widget.

Mirrors the Suite2p run experience: Current-dataset block, output folder
row, Frames & Planes slicing popup (draw_selection_table), a Pipeline
Settings popup with Skip/Run/Force stage columns and modified-orange
tinting against dataclass defaults, a modified-parameters table, and a
green centered Run button. Run spawns the "masknmf" worker task; results
land as suite2p-shaped plane dirs readable by the existing results tooling.
"""

import dataclasses
import math
from pathlib import Path
from typing import Any, Callable

from imgui_bundle import imgui, portable_file_dialogs as pfd

from mbo_utilities.gui._imgui_helpers import (
    PopupAutoSize,
    draw_boxed_label,
    set_tooltip,
)
from mbo_utilities.gui._selection_ui import draw_selection_table, resolve_dim_labels
from mbo_utilities.gui.widgets.pipelines._base import PipelineWidget
from mbo_utilities.gui.widgets.pipelines.settings import (
    _dataset_size_bytes,
    _draw_md_field,
    _format_size,
)
from mbo_utilities.preferences import get_last_dir, set_last_dir
from mbo_utilities.reader import widget_reader_kwargs

# palette matched to the Suite2p settings panel
_TITLE_COLOR = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)
_SUB_COLOR = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)
_DIM_COLOR = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
_WARN_COLOR = imgui.ImVec4(1.0, 0.75, 0.3, 1.0)
_MODIFIED_COLOR = imgui.ImVec4(1.0, 0.72, 0.40, 1.0)

_RUN_W = 220
_BTN_W = 90

# "look at these first" — bold boxed labels, like the suite2p popup's
# _IMPORTANT_FIELDS. Chosen from the reference notebook's guidance: raise
# max_shifts if shift traces clip; mad_correlation_threshold is the main
# signal-vs-noise knob; block_sizes tracks feature size; merge_threshold
# is the notebook's ####-flagged tuning point; filter_sigma drives init.
_IMPORTANT_FIELDS: set[str] = {
    "max_shifts",
    "block_sizes",
    "mad_correlation_threshold",
    "filter_sigma",
    "merge_threshold",
}

# marked stable / do-not-change by the upstream notebook; grouped under a
# collapsed header and flagged in the modified table
_STABLE_FIELDS: set[str] = {
    "sim_conf",
    "spatial_avg_factor",
    "temporal_avg_factor",
    "num_epochs",
    "noise_variance_quantile",
    "mad_threshold",
    "residual_threshold",
    "patch_size",
    "min_peak_distance",
    "deletion_threshold",
    "support_threshold_hi",
    "min_brightness",
    "update_frequency",
    "background_downsampling_factor",
}

# lazy availability check cache
_HAS_MASKNMF: bool | None = None


def _check_masknmf_available() -> bool:
    global _HAS_MASKNMF
    if _HAS_MASKNMF is None:
        import importlib.util

        _HAS_MASKNMF = importlib.util.find_spec("masknmf") is not None
    return _HAS_MASKNMF


_STAGE_FILES = ("motion_correction.hdf5", "compression.hdf5", "demixing_results.hdf5")


def _is_masknmf_plane_dir(p: Path) -> bool:
    return any((p / f).exists() for f in _STAGE_FILES)


def find_masknmf_run(fpath) -> tuple[dict | None, str | None]:
    """Locate a masknmf run at/around ``fpath``.

    Returns ``(params, outdir)``: the settings dict saved in the plane
    dir's ops.npy (None when the run died before stamping it — the stage
    HDF5s still mark the tree as a run), and the folder Run should target
    so stage gating resumes in place. Both None when ``fpath`` is not
    part of a masknmf run.
    """
    if fpath is None:
        return None, None
    if isinstance(fpath, (list, tuple)):
        if not fpath:
            return None, None
        fpath = fpath[0]
    try:
        p = Path(str(fpath))
    except (TypeError, ValueError):
        return None, None
    if not p.exists():
        return None, None
    if p.is_file():
        p = p.parent

    if _is_masknmf_plane_dir(p):
        plane_dir, outdir = p, p.parent
    else:
        plane_dir = None
        try:
            for child in sorted(p.iterdir()):
                if child.is_dir() and _is_masknmf_plane_dir(child):
                    plane_dir, outdir = child, p
                    break
        except (OSError, PermissionError):
            return None, None
        if plane_dir is None:
            return None, None

    params = None
    ops_path = plane_dir / "ops.npy"
    if ops_path.exists():
        try:
            import numpy as np

            ops = np.load(ops_path, allow_pickle=True).item()
            saved = ops.get("masknmf")
            if isinstance(saved, dict):
                params = saved
        except Exception:
            pass
    return params, str(outdir)


def _field_default(obj, name: str):
    for f in dataclasses.fields(obj):
        if f.name == name:
            if f.default is not dataclasses.MISSING:
                return f.default
            if f.default_factory is not dataclasses.MISSING:
                return f.default_factory()
    return None


def _is_default(obj, name: str) -> bool:
    cur = getattr(obj, name)
    default = _field_default(obj, name)
    if isinstance(cur, float) and isinstance(default, (int, float)):
        # imgui.input_float truncates doubles to C float after one frame
        return math.isclose(cur, float(default), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(cur, (tuple, list)) and isinstance(default, (tuple, list)):
        return tuple(cur) == tuple(default)
    return cur == default


def _collect_modified(settings) -> list[tuple[str, Any, Any]]:
    """(field, current, default) rows for every non-default parameter.

    Stage tri-states are excluded — they show in the column-title radios.
    """
    rows = []
    sections = (
        ("reg", settings.registration),
        ("pmd", settings.compression),
        ("demix", settings.demixing),
        ("runtime", settings.runtime),
    )
    for prefix, obj in sections:
        for f in dataclasses.fields(obj):
            if f.name.startswith("do_"):
                continue
            if not _is_default(obj, f.name):
                rows.append(
                    (f"{prefix}.{f.name}", getattr(obj, f.name), _field_default(obj, f.name))
                )
    return rows


class MaskNMFPipelineWidget(PipelineWidget):
    """masknmf processing widget."""

    name = "MaskNMF"
    install_command = (
        "uv pip install git+https://github.com/apasarkar/masknmf-toolbox.git"
    )

    @property
    def is_available(self) -> bool:
        return _check_masknmf_available()

    def __init__(self, parent: Any):
        super().__init__(parent)
        from mbo_utilities.masknmf.params import MasknmfSettings

        self.settings = MasknmfSettings()
        self._outdir = ""
        self._outdir_dialog = None
        self._fix_phase = True
        self._use_fft = True
        self._last_status = ""
        self._show_settings_popup = False
        self._settings_sizer: PopupAutoSize | None = None
        self._show_slice_popup = False
        self._last_fpath = None

    # -- data probes -----------------------------------------------------

    def _array(self):
        iw = getattr(self.parent, "image_widget", None)
        if iw is None or not iw.data:
            return None
        return iw.data[0]

    def _dims(self) -> tuple[int, int, int]:
        """(max_frames, num_planes, num_channels) of the loaded data."""
        max_frames = 1000
        try:
            arr = self._array()
            if arr is not None:
                max_frames = int(getattr(arr, "num_timepoints", None) or arr.shape[0])
        except Exception:
            pass
        num_planes = int(getattr(self.parent, "nz", 0) or 0) or 1
        num_channels = int(getattr(self.parent, "nc", 0) or 0) or 1
        return max_frames, num_planes, num_channels

    def _hydrate_from_run(self, fpath) -> None:
        """Fill parameters + output dir from a previous run's tree
        (suite2p parity: opening results hydrates the Run tab)."""
        from mbo_utilities.masknmf.params import MasknmfSettings

        params, outdir = find_masknmf_run(fpath)
        if outdir is None:
            return
        self._outdir = outdir
        if params is not None:
            self.settings = MasknmfSettings.from_dict(params)
            self._last_status = "Loaded parameters from previous run"
        else:
            self._last_status = "Previous run found (no saved parameters)"

    def _ensure_slice_state(self) -> None:
        """Seed/reset slicing state when the dataset changes (suite2p parity)."""
        fpath = getattr(self.parent, "fpath", None)
        max_frames, num_planes, num_channels = self._dims()
        fpath_changed = fpath != self._last_fpath
        if fpath_changed:
            self._hydrate_from_run(fpath)
        if fpath_changed or getattr(self, "_masknmf_last_max_tp", None) != max_frames:
            self._last_fpath = fpath
            self._masknmf_last_max_tp = max_frames
            self._masknmf_tp_selection = f"1:{max_frames}"
            self._masknmf_tp_parsed = None
            self._masknmf_tp_error = ""
        if getattr(self, "_masknmf_last_num_planes", None) != num_planes:
            self._masknmf_last_num_planes = num_planes
            self._masknmf_z_selection = f"1:{num_planes}"
            self._masknmf_z_start = 1
            self._masknmf_z_stop = num_planes
            self._masknmf_z_step = 1
            self._masknmf_z_error = ""
        if getattr(self, "_masknmf_last_num_channels", None) != num_channels:
            self._masknmf_last_num_channels = num_channels
            self._masknmf_c_selection = f"1:{num_channels}"
            self._masknmf_c_start = 1
            self._masknmf_c_stop = num_channels
            self._masknmf_c_step = 1
            self._masknmf_c_error = ""

    def _selected_planes(self) -> list[int]:
        _, num_planes, _ = self._dims()
        if num_planes <= 1:
            return [1]
        start = getattr(self, "_masknmf_z_start", 1)
        stop = min(getattr(self, "_masknmf_z_stop", num_planes), num_planes)
        step = max(getattr(self, "_masknmf_z_step", 1), 1)
        return list(range(start, stop + 1, step))

    def _selected_channels(self) -> list[int]:
        _, _, num_channels = self._dims()
        if num_channels <= 1:
            return [1]
        start = getattr(self, "_masknmf_c_start", 1)
        stop = min(getattr(self, "_masknmf_c_stop", num_channels), num_channels)
        step = max(getattr(self, "_masknmf_c_step", 1), 1)
        return list(range(start, stop + 1, step))

    def _tp_indices(self) -> list[int] | None:
        parsed = getattr(self, "_masknmf_tp_parsed", None)
        if parsed is not None and getattr(parsed, "final_indices", None):
            return list(parsed.final_indices)
        return None

    # -- draw ------------------------------------------------------------

    def draw_config(self) -> None:
        self._ensure_slice_state()
        imgui.spacing()
        self._draw_dataset_block()
        from mbo_utilities.gui._gpu_status import draw_gpu_status

        draw_gpu_status(self.settings.runtime.device)
        imgui.separator()
        self._draw_output_row()
        self._draw_slice_row()
        imgui.spacing()
        if imgui.button("Pipeline Settings##masknmf_settings", imgui.ImVec2(160, 0)):
            self._show_settings_popup = True
        set_tooltip("Per-stage Skip/Run/Force and parameters.", show_mark=False)
        self._draw_settings_popup()
        imgui.spacing()
        self._draw_modified_table()
        imgui.spacing()
        self._draw_run()

    def _draw_dataset_block(self) -> None:
        imgui.text_colored(_TITLE_COLOR, "Current dataset")
        arr = self._array()
        if arr is None:
            imgui.text_disabled("No dataset loaded.")
            return
        fpath = getattr(self.parent, "fpath", None)
        path_str = str(fpath) if fpath else ""
        short_name = Path(path_str).name or path_str if path_str else "(in-memory)"
        imgui.indent(8)
        imgui.text(f"Name: {short_name}")
        if path_str and imgui.is_item_hovered():
            imgui.set_tooltip(path_str)
        filenames = getattr(arr, "filenames", None) or ([path_str] if path_str else [])
        if filenames:
            imgui.text(f"Size on disk: {_format_size(_dataset_size_bytes(self, filenames))}")
        try:
            shape_text = " × ".join(str(s) for s in arr.shape)
            imgui.text(f"Shape: {shape_text}")
        except Exception:
            pass
        try:
            from mbo_utilities.metadata import get_param

            md = dict(getattr(arr, "metadata", {}) or {})
            md.update(getattr(self.parent, "_custom_metadata", {}) or {})
            _draw_md_field("Frame rate", get_param(md, "fs"), "Hz")
            _, num_planes, _ = self._dims()
            if num_planes > 1:
                _draw_md_field("Z-step", get_param(md, "dz"), "µm")
        except Exception:
            pass
        imgui.unindent(8)

    def _draw_output_row(self) -> None:
        if self._outdir_dialog is not None and self._outdir_dialog.ready():
            result = self._outdir_dialog.result()
            if result:
                self._outdir = result
                set_last_dir("masknmf_outdir", result)
            self._outdir_dialog = None

        imgui.text_colored(_SUB_COLOR, "Output folder")
        imgui.set_next_item_width(max(imgui.get_content_region_avail().x - _BTN_W - 12, 100))
        _, self._outdir = imgui.input_text("##masknmf_outdir", self._outdir)
        set_tooltip("Save path. One zplaneNN dir per plane.", show_mark=False)
        imgui.same_line()
        if imgui.button("Browse##masknmf_outdir_btn", imgui.ImVec2(_BTN_W, 0)):
            start = str(get_last_dir("masknmf_outdir") or Path.home())
            self._outdir_dialog = pfd.select_folder("Select output folder", start)

    def _draw_slice_row(self) -> None:
        max_frames, num_planes, num_channels = self._dims()
        if imgui.button("Set slice##masknmf_slice", imgui.ImVec2(_BTN_W, 0)):
            self._show_slice_popup = True
        set_tooltip("Timepoints / planes / channels to process.", show_mark=False)
        imgui.same_line()
        tp = self._tp_indices()
        n_tp = len(tp) if tp is not None else max_frames
        planes = self._selected_planes()
        channels = self._selected_channels()
        preview = f"{n_tp} frames · {len(planes)} plane(s)"
        if num_channels > 1:
            preview += f" · {len(channels)} channel(s)"
        imgui.text_colored(_DIM_COLOR, preview)

        if self._show_slice_popup:
            imgui.open_popup("Frames & Planes##masknmf_slice")
            self._show_slice_popup = False
        imgui.set_next_window_size(imgui.ImVec2(520, 0), imgui.Cond_.first_use_ever)
        if imgui.begin_popup("Frames & Planes##masknmf_slice"):
            tp_label, z_label, c_label = resolve_dim_labels(self.parent)
            draw_selection_table(
                self,
                max_frames,
                num_planes,
                tp_attr="_masknmf_tp",
                z_attr="_masknmf_z",
                id_suffix="_masknmf",
                num_channels=num_channels,
                c_attr="_masknmf_c",
                tp_label=tp_label,
                z_label=z_label,
                c_label=c_label,
            )
            imgui.spacing()
            _, self._fix_phase = imgui.checkbox(
                "Fix scan phase##masknmf_fixphase", self._fix_phase
            )
            if self._fix_phase:
                imgui.same_line()
                _, self._use_fft = imgui.checkbox("FFT##masknmf_fft", self._use_fft)
            imgui.spacing()
            if imgui.button("Close##masknmf_slice_close", imgui.ImVec2(_BTN_W, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

    # -- settings popup ---------------------------------------------------

    def _mod_push(self, obj, field: str) -> bool:
        if not _is_default(obj, field):
            imgui.push_style_color(imgui.Col_.text, _MODIFIED_COLOR)
            return True
        return False

    @staticmethod
    def _mod_pop(pushed: bool) -> None:
        if pushed:
            imgui.pop_style_color()

    def _param_label(self, obj, field: str, text: str) -> None:
        """suite2p _emp_label pattern: bold boxed label for important
        fields, plain text otherwise; modified-orange tint on both."""
        pushed = self._mod_push(obj, field)
        if field in _IMPORTANT_FIELDS:
            draw_boxed_label(text, font=getattr(self.parent, "_bold_font", None))
        else:
            imgui.text(text)
        self._mod_pop(pushed)

    def _row_tail(self, obj, field: str, label: str, tooltip: str | None) -> None:
        imgui.same_line(0, imgui.get_style().item_inner_spacing.x)
        self._param_label(obj, field, label)
        if tooltip:
            set_tooltip(tooltip, show_mark=False)

    def _f_int(self, obj, field: str, label: str, lo=1, tooltip=None) -> None:
        pushed = self._mod_push(obj, field)
        imgui.set_next_item_width(110)
        _, val = imgui.input_int(f"##masknmf_{field}", getattr(obj, field))
        self._mod_pop(pushed)
        setattr(obj, field, max(lo, val))
        self._row_tail(obj, field, label, tooltip)

    def _f_int2(self, obj, field: str, label: str, lo=1, tooltip=None) -> None:
        pushed = self._mod_push(obj, field)
        imgui.set_next_item_width(110)
        _, v = imgui.input_int2(f"##masknmf_{field}", list(getattr(obj, field)))
        self._mod_pop(pushed)
        setattr(obj, field, (max(lo, v[0]), max(lo, v[1])))
        self._row_tail(obj, field, label, tooltip)

    def _f_float(
        self, obj, field: str, label: str, step=0.05, fmt="%.2f", tooltip=None
    ) -> None:
        pushed = self._mod_push(obj, field)
        imgui.set_next_item_width(110)
        _, val = imgui.input_float(
            f"##masknmf_{field}", getattr(obj, field), step, step * 2, fmt
        )
        self._mod_pop(pushed)
        setattr(obj, field, val)
        self._row_tail(obj, field, label, tooltip)

    def _f_check(self, obj, field: str, label: str, tooltip=None) -> None:
        pushed = self._mod_push(obj, field)
        _, val = imgui.checkbox(f"{label}##masknmf_{field}", getattr(obj, field))
        self._mod_pop(pushed)
        setattr(obj, field, bool(val))
        if tooltip:
            set_tooltip(tooltip, show_mark=False)

    def _stable_header(self, id_suffix: str, body: Callable[[], None]) -> None:
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        imgui.push_style_color(imgui.Col_.text, _DIM_COLOR)
        opened = imgui.collapsing_header(f"Stable — don't change##masknmf_{id_suffix}")
        imgui.pop_style_color()
        set_tooltip(
            "Marked stable by the masknmf authors — leave at defaults unless "
            "you have a reason. Changes show orange here and (stable) in the "
            "modified table.",
            show_mark=False,
        )
        if opened:
            body()

    def _draw_settings_popup(self) -> None:
        popup_title = "MaskNMF Pipeline Settings##masknmf_settings_popup"
        if self._settings_sizer is None:
            self._settings_sizer = PopupAutoSize(popup_title)

        if self._show_settings_popup:
            imgui.open_popup(popup_title)
            self._show_settings_popup = False

        # suite2p popup sizing: always_auto_resize tracks content every
        # frame; constraints cap at the viewport so nothing is cut off
        viewport = imgui.get_main_viewport()
        self._settings_sizer.before_open()
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(min(720.0, viewport.size.x * 0.9), 200.0),
            imgui.ImVec2(viewport.size.x * 0.98, viewport.size.y * 0.98),
        )
        opened, visible = imgui.begin_popup_modal(
            popup_title,
            p_open=True,
            flags=self._settings_sizer.flags(imgui.WindowFlags_.no_saved_settings),
        )
        if not opened:
            return
        try:
            if not visible:
                imgui.close_current_popup()
                return
            columns = (
                ("Registration", self.settings.registration, "do_registration",
                 self._draw_registration_params),
                ("Compression (PMD)", self.settings.compression, "do_compression",
                 self._draw_compression_params),
                ("Demixing", self.settings.demixing, "do_demixing",
                 self._draw_demixing_params),
            )
            avail_w = imgui.get_content_region_avail().x
            col_w = max((avail_w - 16) / 3, 220)
            for i, (title, obj, tri_attr, body) in enumerate(columns):
                if i:
                    imgui.same_line()
                self._draw_stage_column(title, obj, tri_attr, body, col_w)

            imgui.spacing()
            imgui.separator()
            self._draw_runtime_row()
            imgui.spacing()
            imgui.separator()

            # Defaults (orange) / Close (red, right-aligned) — suite2p layout
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.60, 0.35, 0.10, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.70, 0.42, 0.14, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.50, 0.28, 0.08, 1.0))
            if imgui.button("Defaults##masknmf_defaults", imgui.ImVec2(_BTN_W, 0)):
                from mbo_utilities.masknmf.params import MasknmfSettings

                self.settings = MasknmfSettings()
            imgui.pop_style_color(3)
            set_tooltip("Reset every parameter to its default.", show_mark=False)

            imgui.same_line()
            _pad_x = imgui.get_style().window_padding.x
            imgui.set_cursor_pos_x(imgui.get_window_width() - _BTN_W - _pad_x)
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.55, 0.13, 0.13, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.65, 0.18, 0.18, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.45, 0.10, 0.10, 1.0))
            if imgui.button("Close##masknmf_settings_close", imgui.ImVec2(_BTN_W, 0)):
                imgui.close_current_popup()
            imgui.pop_style_color(3)
        finally:
            imgui.end_popup()

    def _draw_stage_column(
        self, title: str, obj, tri_attr: str, body: Callable[[], None], width: float
    ) -> None:
        imgui.begin_child(
            f"##masknmf_col_{tri_attr}",
            imgui.ImVec2(width, 0),
            imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
        )
        # end_child must run even when begin_child returns False
        try:
            imgui.text_colored(_TITLE_COLOR, title)
            value = getattr(obj, tri_attr)
            for i, label in enumerate(("Skip", "Run", "Force")):
                if imgui.radio_button(f"{label}##{tri_attr}", value == i):
                    setattr(obj, tri_attr, i)
                if i < 2:
                    imgui.same_line()
            set_tooltip(
                "Skip: bypass / reuse cached stage output. Run: reuse when "
                "present, else compute. Force: always recompute.",
                show_mark=False,
            )
            imgui.separator()
            skip = getattr(obj, tri_attr) == 0
            if skip:
                imgui.begin_disabled()
            try:
                body()
            finally:
                if skip:
                    imgui.end_disabled()
        finally:
            imgui.end_child()

    def _draw_registration_params(self) -> None:
        reg = self.settings.registration
        pushed = self._mod_push(reg, "strategy")
        idx = 1 if reg.strategy == "pwrigid" else 0
        imgui.set_next_item_width(140)
        changed, idx = imgui.combo(
            "Strategy##masknmf_reg_strategy", idx, ["Rigid", "Piecewise rigid"]
        )
        self._mod_pop(pushed)
        if changed:
            reg.strategy = "pwrigid" if idx == 1 else "rigid"
        self._f_int2(
            reg, "max_shifts", "Max shifts",
            tooltip="Max allowed shift in px (y, x). Raise it if the shift "
                    "traces look clipped/flat at the extremes.",
        )
        if reg.strategy == "pwrigid":
            self._f_int2(reg, "num_blocks", "Blocks")
            self._f_int2(reg, "overlaps", "Overlaps", lo=0)
            self._f_int2(reg, "max_deviation_rigid", "Max deviation", lo=0)

    def _draw_compression_params(self) -> None:
        comp = self.settings.compression
        self._f_check(
            comp, "denoise", "Temporal denoiser",
            tooltip="Train a blind-spot denoiser and re-run PMD.",
        )
        self._f_check(
            comp, "detrend", "Detrend",
            tooltip="Maximin spline detrend sized from the frame rate.",
        )
        self._f_int2(
            comp, "block_sizes", "Block sizes", lo=4,
            tooltip="PMD patch size in px; ~2x the largest feature you "
                    "expect (10-20 for dendrites/spines, 20+ for somata).",
        )
        self._f_int(
            comp, "max_components", "Max components",
            tooltip="Max PMD components per block.",
        )

        def _stable() -> None:
            self._f_int(comp, "sim_conf", "Sim conf")
            self._f_int(comp, "spatial_avg_factor", "Spatial avg")
            self._f_int(comp, "temporal_avg_factor", "Temporal avg")
            if comp.denoise:
                self._f_int(comp, "num_epochs", "Denoiser epochs")
                self._f_float(comp, "noise_variance_quantile", "Noise quantile")

        self._stable_header("pmd_stable", _stable)

    def _draw_demixing_params(self) -> None:
        dmx = self.settings.demixing
        self._f_float(
            dmx, "mad_correlation_threshold", "Correlation thr",
            tooltip="Superpixel seed threshold — the main signal-vs-noise "
                    "knob. Lower finds more, dimmer cells.",
        )
        self._f_float(
            dmx, "filter_sigma", "Highpass sigma", step=0.5, fmt="%.1f",
            tooltip="Spatial highpass width (px) for the init passes.",
        )
        self._f_float(
            dmx, "merge_threshold", "Merge thr",
            tooltip="Temporal correlation above which overlapping "
                    "components merge.",
        )
        self._f_int(
            dmx, "maxiter", "NMF iterations",
            tooltip="HALS iterations per pass.",
        )
        pushed = self._mod_push(dmx, "sign")
        sign_idx = ("positive", "negative", "unconstrained").index(dmx.sign)
        imgui.set_next_item_width(140)
        changed, sign_idx = imgui.combo(
            "Signal sign##masknmf_sign", sign_idx,
            ["Positive", "Negative", "Unconstrained"],
        )
        self._mod_pop(pushed)
        if changed:
            dmx.sign = ("positive", "negative", "unconstrained")[sign_idx]
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.collapsing_header("Advanced##masknmf_demix_adv"):
            self._f_int(dmx, "filtered_passes", "Filtered passes")
            self._f_int(dmx, "unfiltered_passes", "Unfiltered passes")
            self._f_int(dmx, "ring_radius", "Ring radius")
            self._f_float(dmx, "support_threshold_lo", "Support thr")
            self._f_float(dmx, "unfiltered_support_lo", "Unfiltered support")
            self._f_float(dmx, "merge_overlap_threshold", "Merge overlap")
            self._f_check(dmx, "reassign_background", "Reassign background")

        def _stable() -> None:
            self._f_float(dmx, "mad_threshold", "MAD thr", step=0.1, fmt="%.1f")
            self._f_float(dmx, "residual_threshold", "Residual thr")
            self._f_int2(dmx, "patch_size", "Patch size", lo=8)
            self._f_int(dmx, "min_peak_distance", "Min peak dist")
            self._f_float(dmx, "deletion_threshold", "Deletion thr")
            self._f_float(dmx, "support_threshold_hi", "Support thr hi")
            self._f_float(dmx, "min_brightness", "Min brightness", step=0.1, fmt="%.1f")
            self._f_int(dmx, "update_frequency", "Update freq")
            self._f_int(dmx, "background_downsampling_factor", "Bg downsample")

        self._stable_header("demix_stable", _stable)

    def _draw_runtime_row(self) -> None:
        rt = self.settings.runtime
        imgui.text_colored(_SUB_COLOR, "Runtime")
        pushed = self._mod_push(rt, "device")
        dev_idx = ("auto", "cuda", "cpu").index(rt.device)
        imgui.set_next_item_width(110)
        changed, dev_idx = imgui.combo(
            "Device##masknmf_device", dev_idx, ["Auto", "CUDA", "CPU"]
        )
        self._mod_pop(pushed)
        if changed:
            rt.device = ("auto", "cuda", "cpu")[dev_idx]
        if rt.device == "cpu":
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(
                _WARN_COLOR, "masknmf superpixel init currently requires CUDA"
            )
            imgui.pop_text_wrap_pos()
        pushed = self._mod_push(rt, "frame_batch_size")
        imgui.set_next_item_width(110)
        _, rt.frame_batch_size = imgui.input_int(
            "Frame batch##masknmf_batch", rt.frame_batch_size
        )
        self._mod_pop(pushed)
        rt.frame_batch_size = max(10, rt.frame_batch_size)
        imgui.same_line()
        pushed = self._mod_push(rt, "exclude_border_radius")
        imgui.set_next_item_width(110)
        _, rt.exclude_border_radius = imgui.input_int(
            "Exclude border px##masknmf_border", rt.exclude_border_radius
        )
        self._mod_pop(pushed)
        rt.exclude_border_radius = max(0, rt.exclude_border_radius)
        pushed = self._mod_push(rt, "keep_bin")
        _, rt.keep_bin = imgui.checkbox(
            "Write registered data.bin##masknmf_keepbin", rt.keep_bin
        )
        self._mod_pop(pushed)
        set_tooltip("Registered movie as suite2p binary.", show_mark=False)
        imgui.same_line()
        pushed = self._mod_push(rt, "keep_raw")
        _, rt.keep_raw = imgui.checkbox("Keep data_raw.bin##masknmf_keepraw", rt.keep_raw)
        self._mod_pop(pushed)

    def _draw_modified_table(self) -> None:
        mods = _collect_modified(self.settings)
        imgui.text(f"Modified parameters ({len(mods)})")
        if not mods:
            imgui.text_disabled("All parameters at defaults")
            return
        mod_h = min(160, imgui.get_frame_height_with_spacing() * (len(mods) + 2))
        if imgui.begin_child(
            "##masknmf_mod_params", imgui.ImVec2(-1, mod_h), imgui.ChildFlags_.borders
        ):
            flags = (
                imgui.TableFlags_.row_bg
                | imgui.TableFlags_.borders_inner_h
                | imgui.TableFlags_.sizing_stretch_prop
            )
            if imgui.begin_table("##masknmf_mod_tbl", 3, flags):
                imgui.table_setup_column("Parameter", imgui.TableColumnFlags_.width_stretch, 4.0)
                imgui.table_setup_column("Current", imgui.TableColumnFlags_.width_stretch, 2.5)
                imgui.table_setup_column("Default", imgui.TableColumnFlags_.width_stretch, 2.5)
                imgui.table_headers_row()
                for field, cur, default in mods:
                    cur_s = f"{cur:.3g}" if isinstance(cur, float) else str(cur)
                    def_s = f"{default:.3g}" if isinstance(default, float) else str(default)
                    stable = field.split(".", 1)[-1] in _STABLE_FIELDS
                    imgui.table_next_row()
                    imgui.table_set_column_index(0)
                    imgui.text_colored(_MODIFIED_COLOR, field)
                    if stable:
                        imgui.same_line()
                        imgui.text_colored(_WARN_COLOR, "(stable)")
                        set_tooltip(
                            "Marked stable by the masknmf authors — change "
                            "only with reason.",
                            show_mark=False,
                        )
                    imgui.table_set_column_index(1)
                    imgui.text(cur_s)
                    imgui.table_set_column_index(2)
                    imgui.text_disabled(def_s)
                imgui.end_table()
        imgui.end_child()

    def _draw_run(self) -> None:
        planes = self._selected_planes()
        fpath = getattr(self.parent, "fpath", None)
        has_save_path = bool(self._outdir)
        ready = has_save_path and bool(fpath) and bool(planes)

        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.13, 0.55, 0.13, 1.0))
        imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.18, 0.65, 0.18, 1.0))
        imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.1, 0.45, 0.1, 1.0))
        if not ready:
            imgui.begin_disabled()
        run_avail = imgui.get_content_region_avail().x
        if run_avail > _RUN_W:
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (run_avail - _RUN_W) * 0.5)
        clicked = imgui.button("Run MaskNMF", imgui.ImVec2(_RUN_W, 0))
        if not ready:
            imgui.end_disabled()
        imgui.pop_style_color(3)

        if not ready and imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled):
            imgui.set_tooltip(
                "Load a dataset and set the output folder"
                if not has_save_path
                else "Load a dataset"
            )
        if self._last_status:
            imgui.text_colored(_DIM_COLOR, self._last_status)

        if clicked and ready:
            self._submit(planes)

    def _submit(self, planes: list[int]) -> None:
        from mbo_utilities.gui.widgets.process_manager import get_process_manager

        pm = get_process_manager()
        channels = self._selected_channels()
        multi_channel = len(channels) > 1
        # multi-channel SOURCES need the channel selection even when a
        # single channel is picked, else the bin write takes every channel
        has_channels = self._dims()[2] > 1
        tp_indices = self._tp_indices()

        started = []
        for channel in channels:
            output_dir = (
                str(Path(self._outdir) / f"ch{channel:02d}")
                if multi_channel
                else self._outdir
            )
            args = {
                "input_path": str(self.parent.fpath),
                "reader_kwargs": widget_reader_kwargs(
                    getattr(self.parent, "image_widget", None)
                ),
                "output_dir": output_dir,
                "planes": planes,
                "settings": self.settings.to_dict(),
                "fix_phase": self._fix_phase,
                "use_fft": self._use_fft,
                "tp_indices": tp_indices,
                "selected_planes_0based": [p - 1 for p in planes],
                "channel": channel if (multi_channel or has_channels) else None,
                "custom_metadata": dict(getattr(self.parent, "_custom_metadata", {})),
            }
            if len(planes) == 1:
                description = f"MaskNMF plane{planes[0]:02d}"
            else:
                description = f"MaskNMF: {len(planes)} plane(s)"
            if multi_channel:
                description += f" ch{channel}"

            pid = pm.spawn(
                task_type="masknmf",
                args=args,
                description=description,
                output_path=output_dir,
            )
            if pid:
                started.append(pid)

        if started:
            self._last_status = f"Started {len(started)} job(s) (PID {started[0]})"
        else:
            self._last_status = "Failed to start worker."

    def cleanup(self) -> None:
        self._outdir_dialog = None
