import socket
import threading
from pathlib import Path

from imgui_bundle import (
    hello_imgui,
    imgui,
    imgui_ctx,
    portable_file_dialogs as pfd,
    icons_fontawesome_6 as fa,
)
from mbo_utilities.gui import _setup  # triggers setup on import
from mbo_utilities.preferences import (
    get_default_open_dir,
    set_last_dir,
    add_recent_file,
    get_gpu_index,
    set_gpu_index,
    get_debug_logging,
    set_debug_logging,
)
from mbo_utilities.install import Status, check_installation, gpu_summary
from mbo_utilities.gui._files import NATIVE_DIALOGS, no_dialog_hint

# re-export for backwards compatibility
setup_imgui = _setup.setup_imgui
__all__ = ["FileDialog", "setup_imgui"]

# dark theme
COL_BG = imgui.ImVec4(0.11, 0.11, 0.12, 1.0)
COL_BG_CARD = imgui.ImVec4(0.16, 0.16, 0.17, 1.0)
COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
COL_ACCENT_HOVER = imgui.ImVec4(0.25, 0.55, 0.90, 1.0)
COL_ACCENT_ACTIVE = imgui.ImVec4(0.15, 0.45, 0.80, 1.0)
COL_TEXT = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
COL_TEXT_DIM = imgui.ImVec4(0.75, 0.75, 0.77, 1.0)
COL_BORDER = imgui.ImVec4(0.35, 0.35, 0.37, 0.7)
COL_SECONDARY = imgui.ImVec4(0.35, 0.35, 0.37, 1.0)
COL_SECONDARY_HOVER = imgui.ImVec4(0.42, 0.42, 0.44, 1.0)
COL_SECONDARY_ACTIVE = imgui.ImVec4(0.28, 0.28, 0.30, 1.0)
COL_OK = imgui.ImVec4(0.4, 1.0, 0.4, 1.0)
COL_WARN = imgui.ImVec4(1.0, 0.8, 0.2, 1.0)
COL_ERR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
COL_NA = imgui.ImVec4(0.5, 0.5, 0.5, 1.0)
# the dependency popup sits below the window and the formats card in tone
COL_BG_POPUP = imgui.ImVec4(0.06, 0.06, 0.07, 1.0)
COL_ROW_ALT = imgui.ImVec4(0.10, 0.10, 0.11, 1.0)

# launcher table rows, in order; the CLI prints every feature
_DEP_ROWS = ("PyTorch", "LBM-Suite2p-Python", "Cellpose", "MaskNMF", "CuPy", "Rastermap", "Napari")
_STATUS_GLYPH = {
    Status.OK: (fa.ICON_FA_CIRCLE_CHECK, COL_OK),
    Status.WARN: (fa.ICON_FA_CIRCLE_EXCLAMATION, COL_WARN),
    Status.ERROR: (fa.ICON_FA_CIRCLE_XMARK, COL_ERR),
    Status.MISSING: (fa.ICON_FA_CIRCLE_MINUS, COL_NA),
}


def _get_install_source() -> str:
    """get install source label: 'PyPI', git branch name, or 'editable'."""
    try:
        import importlib.metadata
        import json

        dist = importlib.metadata.distribution("mbo-utilities")

        # check direct_url.json (pip install from git url or editable)
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            info = json.loads(direct_url_text)
            vcs_info = info.get("vcs_info", {})
            if vcs_info:
                branch = vcs_info.get("requested_revision")
                if branch:
                    return branch
                commit = vcs_info.get("commit_id", "")
                return commit[:8] if commit else "git"
            if info.get("dir_info"):
                return _git_branch() or "editable"
            return "local"

        # editable egg-info installs (uv run, pip install -e .)
        if hasattr(dist, "_path") and ".egg-info" in str(dist._path):
            return _git_branch() or "editable"

        return "PyPI"
    except Exception:
        return ""


def _git_branch() -> str:
    """get current git branch of the package source, or empty string."""
    try:
        import subprocess
        from pathlib import Path
        import mbo_utilities

        pkg_dir = Path(mbo_utilities.__file__).parent.parent
        if not (pkg_dir / ".git").exists():
            return ""
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=str(pkg_dir), timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def push_button_style(primary=True):
    if primary:
        imgui.push_style_color(imgui.Col_.button, COL_ACCENT)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_ACCENT_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_ACCENT_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    else:
        imgui.push_style_color(imgui.Col_.button, COL_SECONDARY)
        imgui.push_style_color(imgui.Col_.button_hovered, COL_SECONDARY_HOVER)
        imgui.push_style_color(imgui.Col_.button_active, COL_SECONDARY_ACTIVE)
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)


def pop_button_style():
    imgui.pop_style_var(2)
    imgui.pop_style_color(4)


# Tooltips and copy in this dialog must respect the dialog's 340 px width.
# wrapped_tooltip() does NOT auto-wrap, so anything longer than a few
# words runs off the side of the screen on this layout. The helpers below
# always wrap to ~30 em (roughly the dialog width minus margins).
_TOOLTIP_WRAP_EM = 24.0


def wrapped_tooltip(text: str, wrap_em: float = _TOOLTIP_WRAP_EM) -> None:
    """Tooltip whose text wraps to ``wrap_em`` em units wide."""
    imgui.begin_tooltip()
    try:
        imgui.push_text_wrap_pos(hello_imgui.em_size(wrap_em))
        imgui.text_unformatted(text)
        imgui.pop_text_wrap_pos()
    finally:
        imgui.end_tooltip()


def icon_button(icon: str, label: str, size: imgui.ImVec2, tooltip: str = "") -> bool:
    """
    Draw a styled icon button with MBO theme.

    Dark gray background with blue outline, blue icon+text, hover effect.

    Parameters
    ----------
    icon : str
        FontAwesome 6 icon character (e.g., fa.ICON_FA_FOLDER_OPEN)
    label : str
        Button label text
    size : imgui.ImVec2
        Button size
    tooltip : str
        Tooltip text shown on hover

    Returns
    -------
    bool
        True if button was clicked
    """
    # Style: dark gray bg, blue border, blue text
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.18, 0.18, 0.20, 1.0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.22, 0.22, 0.25, 1.0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.15, 0.15, 0.17, 1.0))
    imgui.push_style_color(imgui.Col_.text, COL_ACCENT)
    imgui.push_style_color(imgui.Col_.border, COL_ACCENT)
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    imgui.push_style_var(imgui.StyleVar_.frame_border_size, 1.5)

    # Combine icon and label
    button_text = f"{icon}  {label}"
    clicked = imgui.button(button_text, size)

    # Show tooltip on hover (wrapped so it fits the dialog width)
    if tooltip and imgui.is_item_hovered():
        wrapped_tooltip(tooltip)

    imgui.pop_style_var(2)
    imgui.pop_style_color(5)

    return clicked


class FileDialog:
    def __init__(self):
        self.selected_path = None
        self._open_multi = None
        self._select_folder = None
        # the typed route: works where no native dialog can be drawn
        self._typed_path = ""
        self._typed_status = ""
        self._widget_enabled = True
        self.metadata_only = False
        self.split_rois = False
        self._default_dir = str(get_default_open_dir())
        self._install_source = _get_install_source()
        # cached install status (computed in background)
        self._install_status = None
        self._check_thread = None
        self._show_deps_popup = False

        # GUI Modes (pollen calibration auto-detected, not user-selectable)
        self.gui_modes = ["Fastplotlib viewer (default)", "Napari viewer"]
        self.selected_mode_index = 0

        # GPU adapter selection. -1 == wgpu auto-pick; otherwise an index
        # into self._gpu_adapter_labels. Lazily populated on first render
        # so importing FileDialog doesn't force fastplotlib import.
        self._gpu_adapters = None  # list[wgpu adapter]
        self._gpu_adapter_labels: list[str] = []
        # Seed from persisted preferences so re-opens remember the user's choice.
        self.selected_gpu_index: int = get_gpu_index()
        self.debug_logging: bool = get_debug_logging()
        self._show_options_popup: bool = False
        # nvidia-smi compute devices for the Compute GPU combo; refreshed once
        # per popup-open (subprocess), not per frame.
        self._compute_devices: list = []

        # start dependency check immediately in background
        self._start_dependency_check()

    def _ensure_gpu_list(self) -> None:
        """Read the pre-warmed adapter cache. NEVER call
        ``fpl.enumerate_adapters()`` from here — it initializes wgpu,
        which clobbers GLFW's WGL current-context (Glfw Error 65544)
        and makes the host window flicker. The cache is primed in
        ``_run_gui_impl`` before ``immapp.run`` takes over the GL
        context.
        """
        if self._gpu_adapters is not None:
            return
        from mbo_utilities.gui._gpu_cache import get_adapters
        self._gpu_adapters = list(get_adapters())
        labels = ["auto (default)"]
        for i, a in enumerate(self._gpu_adapters):
            info = getattr(a, "info", {}) or {}
            name = info.get("device", info.get("description", f"adapter {i}"))
            type_ = info.get("adapter_type", info.get("device_type", "?"))
            labels.append(f"{i}: {name} [{type_}]")
        self._gpu_adapter_labels = labels

    def _draw_options_popup(self) -> None:
        """Modal popup: GPU adapter selector + debug logging flag.

        Both settings are persisted to ~/.mbo/settings/preferences.json on
        change so the choice survives restarts.
        """
        if self._show_options_popup:
            try:
                from mbo_utilities.gpu import gpu_devices
                self._compute_devices = gpu_devices()
            except Exception:
                self._compute_devices = []
            # re-read prefs on open; they may have changed elsewhere
            from mbo_utilities.gui._options_popup import sync_memory_options
            sync_memory_options(self)
            imgui.open_popup("##options_popup")
            self._show_options_popup = False

        popup_flags = imgui.WindowFlags_.always_auto_resize
        if not imgui.begin_popup("##options_popup", popup_flags):
            return

        try:
            imgui.text_colored(COL_ACCENT, f"{fa.ICON_FA_GEARS}  Options")
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            self._ensure_gpu_list()

            imgui.text_colored(COL_TEXT_DIM, "GPU adapter")
            if imgui.is_item_hovered():
                wrapped_tooltip(
                    "Pick which GPU to render with. 'auto' lets wgpu "
                    "choose (usually the DiscreteGPU). Takes effect on "
                    "the next dataset you open."
                )
            imgui.set_next_item_width(hello_imgui.em_size(20))
            ui_idx = self.selected_gpu_index + 1  # 0 == "auto"
            changed, new_ui_idx = imgui.combo(
                "##gpu_adapter", ui_idx, self._gpu_adapter_labels
            )
            if changed:
                self.selected_gpu_index = new_ui_idx - 1
                set_gpu_index(self.selected_gpu_index)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # Compute GPU (suite2p / cellpose) — persisted + applied to new runs.
            from mbo_utilities.gui._options_popup import (
                compute_gpu_options,
                compute_gpu_current_index,
                apply_compute_gpu,
                draw_memory_options,
            )
            imgui.text_colored(COL_TEXT_DIM, "Compute GPU (suite2p / cellpose)")
            if imgui.is_item_hovered():
                wrapped_tooltip(
                    "Which GPU suite2p and cellpose run on. 'auto' uses all "
                    "visible GPUs, 'cpu' forces CPU. Saved and applied to the "
                    "next run."
                )
            values, labels = compute_gpu_options(self._compute_devices)
            sel = compute_gpu_current_index(values)
            imgui.set_next_item_width(hello_imgui.em_size(20))
            c_changed, new_c = imgui.combo("##compute_gpu", sel, labels)
            if c_changed and 0 <= new_c < len(values):
                apply_compute_gpu(values[new_c])

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            changed, new_debug = imgui.checkbox("Debug logging", self.debug_logging)
            if imgui.is_item_hovered():
                wrapped_tooltip(
                    "Verbose console logs (per-read timings, zarr chunk "
                    "shapes, etc.). Same effect as launching with "
                    "MBO_DEBUG=1."
                )
            if changed:
                self.debug_logging = new_debug
                set_debug_logging(self.debug_logging)
                # Apply immediately so the dialog itself sees the new level.
                try:
                    import logging
                    from mbo_utilities import log as _mbo_log
                    _mbo_log.set_global_level(
                        logging.DEBUG if self.debug_logging else logging.INFO
                    )
                except Exception:
                    pass

            draw_memory_options(self, tooltip=wrapped_tooltip)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            if imgui.button("Close", imgui.ImVec2(hello_imgui.em_size(6), 0)):
                imgui.close_current_popup()
        finally:
            imgui.end_popup()

    @property
    def widget_enabled(self):
        return self._widget_enabled

    @widget_enabled.setter
    def widget_enabled(self, value):
        self._widget_enabled = value

    def _save_gui_preferences(self):
        pass

    def _start_dependency_check(self):
        """Probe dependencies once, on a thread, from the cache when it is fresh."""
        if self._check_thread is not None:
            return

        def _run_check():
            try:
                from mbo_utilities.env_cache import (
                    build_full_cache_with_install_status,
                    get_cached_install_status,
                    save_cache,
                )

                cached = get_cached_install_status()
                if cached is None:
                    save_cache(build_full_cache_with_install_status())
                    cached = get_cached_install_status()
                self._install_status = cached or check_installation()
            except Exception:
                self._install_status = check_installation()

        self._check_thread = threading.Thread(target=_run_check, daemon=True)
        self._check_thread.start()

    def _dep_rows(self) -> list:
        """Features the launcher table shows, in ``_DEP_ROWS`` order."""
        found = map(self._install_status.feature, _DEP_ROWS)
        return [f for f in found if f is not None]

    def _draw_version_status(self):
        """Draw version with install source inline."""
        from mbo_utilities import __version__

        source = self._install_source
        label = f"v{__version__}"
        if source:
            label += f" ({source})"
        imgui.text_colored(COL_TEXT_DIM, label)

    def _draw_formats_card_content(self):
        """Draw supported formats - always shown immediately."""
        # version line (shows ? while loading)
        self._draw_version_status()

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

        # supported formats section - always visible
        imgui.text_colored(COL_ACCENT, "Supported Formats")
        imgui.same_line()
        push_button_style(primary=False)
        if imgui.small_button(f"{fa.ICON_FA_BOOK}  docs"):
            import webbrowser
            webbrowser.open("https://millerbrainobservatory.github.io/mbo_utilities/file_formats.html")
        pop_button_style()
        if imgui.is_item_hovered():
            wrapped_tooltip("Open documentation in browser")

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.1))

        # calculate table width to fit content
        col1_width = hello_imgui.em_size(6)
        col2_width = hello_imgui.em_size(9)
        table_width = col1_width + col2_width

        table_flags = (
            imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.no_host_extend_x
        )
        if imgui.begin_table("##array_types", 2, table_flags, imgui.ImVec2(table_width, 0)):
            imgui.table_setup_column("Format", imgui.TableColumnFlags_.width_fixed, col1_width)
            imgui.table_setup_column("Extensions", imgui.TableColumnFlags_.width_fixed, col2_width)
            imgui.table_headers_row()

            array_types = [
                ("ScanImage", ".tif, .tiff"),
                ("TIFF", ".tif, .tiff"),
                ("Zarr", ".zarr/"),
                ("HDF5", ".h5, .hdf5"),
                ("Femtonics", ".mesc"),
                ("Suite2p", ".bin, ops.npy"),
                ("NumPy", ".npy"),
            ]
            for name, ext in array_types:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text(name)
                imgui.table_next_column()
                imgui.text_colored(COL_TEXT_DIM, ext)
            imgui.end_table()

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

        # dependency status - small inline section
        self._draw_dependency_status_line()

    def _draw_dependency_status_line(self):
        """One-line summary that opens the dependency table."""
        if self._install_status is None:
            imgui.text_colored(COL_TEXT_DIM, f"{fa.ICON_FA_CIRCLE_NOTCH}  checking dependencies...")
            return
        rows = self._dep_rows()
        ready = sum(f.status is Status.OK for f in rows)
        issues = sum(f.status in (Status.WARN, Status.ERROR) for f in rows)
        icon, color = _STATUS_GLYPH[Status.WARN if issues else Status.OK]
        imgui.text_colored(color, icon)
        imgui.same_line()
        label = f"{ready} of {len(rows)} ready" + (f", {issues} to fix" if issues else "")
        push_button_style(primary=False)
        if imgui.small_button(f"dependencies: {label}"):
            self._show_deps_popup = True
        pop_button_style()
        if imgui.is_item_hovered():
            wrapped_tooltip("Pipelines and GPU backends the GUI can use")
        self._draw_dependency_popup(rows)

    def _draw_dependency_popup(self, rows: list):
        """Table of the GUI's optional packages: version, compute device, fix."""
        if self._show_deps_popup:
            imgui.open_popup("##deps_popup")
            self._show_deps_popup = False
        # darker than the formats card so the two surfaces read as different
        imgui.push_style_color(imgui.Col_.popup_bg, COL_BG_POPUP)
        imgui.push_style_color(imgui.Col_.border, COL_ACCENT)
        imgui.push_style_color(imgui.Col_.table_row_bg_alt, COL_ROW_ALT)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 1.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(1.0, 0.8))
        try:
            if imgui.begin_popup("##deps_popup", imgui.WindowFlags_.always_auto_resize):
                self._draw_dependency_table(rows)
                imgui.end_popup()
        finally:
            imgui.pop_style_var(2)
            imgui.pop_style_color(3)

    def _draw_dependency_table(self, rows: list):
        """Header and table; the table sizes itself so nothing is clipped."""
        imgui.text_colored(COL_ACCENT, "Dependencies")
        gpu = gpu_summary(self._install_status.cuda_info)
        imgui.text_colored(COL_TEXT_DIM, f"{fa.ICON_FA_MICROCHIP}  {gpu}")
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
        flags = (
            imgui.TableFlags_.borders_inner_v
            | imgui.TableFlags_.row_bg
            | imgui.TableFlags_.sizing_fixed_fit
            | imgui.TableFlags_.no_host_extend_x
        )
        if imgui.begin_table("##deps", 4, flags):
            for title in ("Package", "Version", "Device", ""):
                imgui.table_setup_column(title, imgui.TableColumnFlags_.width_fixed)
            imgui.table_headers_row()
            for f in rows:
                self._draw_dependency_row(f)
            imgui.end_table()

    def _draw_dependency_row(self, f):
        """One feature: the row carries the tooltip, the last cell copies the fix."""
        icon, color = _STATUS_GLYPH[f.status]
        installed = f.status is not Status.MISSING
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.selectable(
            f"##dep-{f.name}",
            False,
            imgui.SelectableFlags_.span_all_columns | imgui.SelectableFlags_.allow_overlap,
        )
        hovered = imgui.is_item_hovered()
        imgui.same_line(0, 0)
        imgui.text_colored(color, icon)
        imgui.same_line()
        imgui.text_colored(COL_TEXT if installed else COL_NA, f.name)
        imgui.table_next_column()
        imgui.text_colored(COL_TEXT_DIM, f.version if installed else "-")
        imgui.table_next_column()
        if installed and f.gpu_ok is not None:
            imgui.text_colored(COL_OK if f.gpu_ok else COL_WARN, "GPU" if f.gpu_ok else "CPU")
        imgui.table_next_column()
        fixable = bool(f.hint) and f.status is not Status.OK
        if fixable:
            push_button_style(primary=False)
            if imgui.small_button(f"{fa.ICON_FA_COPY}##fix-{f.name}"):
                imgui.set_clipboard_text(f.hint)
            pop_button_style()
            if imgui.is_item_hovered():
                wrapped_tooltip(f"copy: {f.hint}")
                hovered = False
        if hovered:
            lines = [f.purpose]
            if f.message and f.message != "ready":
                lines.append(f.message)
            if fixable:
                lines.append(f"fix: {f.hint}")
            wrapped_tooltip("\n".join(lines))

    def _open_typed_path(self) -> None:
        """Accept the typed path the way a dialog result is accepted."""
        raw = self._typed_path.strip().strip('"')
        if not raw:
            self._typed_status = "type a file or folder path"
            return
        p = Path(raw).expanduser()
        if not p.exists():
            self._typed_status = f"not found on this machine: {p}"
            return
        self._typed_status = ""
        self.selected_path = str(p)
        kind = "folder" if p.is_dir() else "file"
        add_recent_file(self.selected_path, file_type=kind)
        set_last_dir(f"open_{kind}", self.selected_path)
        self._save_gui_preferences()
        hello_imgui.get_runner_params().app_shall_exit = True

    def _center_text(self, text, color=None):
        """Draw centered text."""
        avail_w = imgui.get_content_region_avail().x
        text_sz = imgui.calc_text_size(text)
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - text_sz.x) * 0.5)
        if color:
            imgui.text_colored(color, text)
        else:
            imgui.text(text)

    def _center_widget(self, widget_width):
        """Set cursor to center a widget of given width."""
        avail_w = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - widget_width) * 0.5)

    def render(self):
        # global style
        imgui.push_style_color(imgui.Col_.window_bg, COL_BG)
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.text, COL_TEXT)
        imgui.push_style_color(imgui.Col_.border, COL_BORDER)
        imgui.push_style_color(imgui.Col_.separator, imgui.ImVec4(0.35, 0.35, 0.37, 0.6))
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.22, 0.22, 0.23, 1.0))
        imgui.push_style_color(imgui.Col_.frame_bg_hovered, imgui.ImVec4(0.28, 0.28, 0.29, 1.0))
        imgui.push_style_color(imgui.Col_.check_mark, COL_ACCENT)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(1.0, 0.8))
        imgui.push_style_var(imgui.StyleVar_.frame_padding, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.item_spacing, hello_imgui.em_to_vec2(0.6, 0.4))
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)

        with imgui_ctx.begin_child("##main", size=imgui.ImVec2(0, 0), window_flags=imgui.WindowFlags_.no_scrollbar):
            imgui.push_id("pfd")

            # header
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            self._center_text("Miller Brain Observatory", COL_ACCENT)
            self._center_text("Data Preview & Utilities", COL_TEXT_DIM)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            imgui.separator()
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            # action buttons - use full available width minus padding
            avail_w = imgui.get_content_region_avail().x
            btn_w = min(avail_w - hello_imgui.em_size(2), hello_imgui.em_size(16))
            btn_h = hello_imgui.em_size(1.8)

            # Mode Selector
            self._center_widget(btn_w)
            imgui.set_next_item_width(btn_w)

            # Simple combo
            # ret, idx = imgui.combo("##mode", current_item, items)
            _changed, self.selected_mode_index = imgui.combo(
                "##mode",
                self.selected_mode_index,
                self.gui_modes
            )
            if imgui.is_item_hovered():
                wrapped_tooltip(f"Select Application: {self.gui_modes[self.selected_mode_index]}")

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            # native dialogs, disabled where nothing can draw one (a linux
            # box without zenity/kdialog); the typed field below always works
            if not NATIVE_DIALOGS:
                imgui.begin_disabled()
            self._center_widget(btn_w)
            if icon_button(
                fa.ICON_FA_FILE_IMAGE,
                "Open File(s)",
                imgui.ImVec2(btn_w, btn_h),
                "Select one or more image files" if NATIVE_DIALOGS else no_dialog_hint()
            ):
                self._open_multi = pfd.open_file(
                    "Select files",
                    self._default_dir,
                    ["Image Files", "*.tif *.tiff *.zarr *.npy *.bin",
                     "All Files", "*"],
                    pfd.opt.multiselect
                )

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            self._center_widget(btn_w)
            if icon_button(
                fa.ICON_FA_FOLDER_OPEN,
                "Select Folder",
                imgui.ImVec2(btn_w, btn_h),
                "Select folder with image data" if NATIVE_DIALOGS else no_dialog_hint()
            ):
                self._select_folder = pfd.select_folder("Select folder", self._default_dir)
            if not NATIVE_DIALOGS:
                imgui.end_disabled()

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))

            # typed path: file or folder, opened on enter or the button
            self._center_widget(btn_w)
            imgui.set_next_item_width(btn_w - hello_imgui.em_size(3.2))
            entered, self._typed_path = imgui.input_text_with_hint(
                "##typed_path",
                "or type a path",
                self._typed_path,
                imgui.InputTextFlags_.enter_returns_true,
            )
            if imgui.is_item_hovered():
                wrapped_tooltip(
                    f"A file or folder on {socket.gethostname()}, where this "
                    "process runs. Enter opens it."
                )
            imgui.same_line()
            if imgui.button("Open", imgui.ImVec2(hello_imgui.em_size(3), 0)) or entered:
                self._open_typed_path()
            if self._typed_status:
                self._center_text(self._typed_status, COL_TEXT_DIM)

            imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))

            # card - use available width minus small margin
            avail_w = imgui.get_content_region_avail().x
            card_w = avail_w - hello_imgui.em_size(1)
            self._center_widget(card_w)

            imgui.push_style_color(imgui.Col_.child_bg, COL_BG_CARD)
            imgui.push_style_var(imgui.StyleVar_.child_rounding, 6.0)
            imgui.push_style_var(imgui.StyleVar_.cell_padding, hello_imgui.em_to_vec2(0.4, 0.2))

            # auto-resize height to content, no scrollbar
            child_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
            window_flags = imgui.WindowFlags_.no_scrollbar

            with imgui_ctx.begin_child("##formats", size=imgui.ImVec2(card_w, 0), child_flags=child_flags, window_flags=window_flags):
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
                imgui.indent(hello_imgui.em_size(0.6))

                self._draw_formats_card_content()

                imgui.unindent(hello_imgui.em_size(0.6))
                imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

            imgui.pop_style_var(2)
            imgui.pop_style_color()

            # Options popup is drawn here but only renders when opened
            # via the button on the bottom row.
            self._draw_options_popup()

            # file/folder completion
            if self._open_multi and self._open_multi.ready():
                self.selected_path = self._open_multi.result()
                if self.selected_path:
                    for p in (self.selected_path if isinstance(self.selected_path, list) else [self.selected_path]):
                        add_recent_file(p, file_type="file")
                        set_last_dir("open_file", p)
                    self._save_gui_preferences()
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._open_multi = None
            if self._select_folder and self._select_folder.ready():
                self.selected_path = self._select_folder.result()
                if self.selected_path:
                    add_recent_file(self.selected_path, file_type="folder")
                    set_last_dir("open_folder", self.selected_path)
                    self._save_gui_preferences()
                    hello_imgui.get_runner_params().app_shall_exit = True
                self._select_folder = None

            # Options + Quit on a centered row so both remain visible
            # regardless of how tall the formats card grew.
            imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
            opt_w = hello_imgui.em_size(7)
            quit_w = hello_imgui.em_size(6)
            spacing = imgui.get_style().item_spacing.x
            row_w = opt_w + quit_w + spacing
            self._center_widget(row_w)
            btn_h = hello_imgui.em_size(1.5)
            # Options uses secondary style for visual parity with Quit
            push_button_style(primary=False)
            if imgui.button(f"{fa.ICON_FA_GEARS}  Options", imgui.ImVec2(opt_w, btn_h)):
                self._show_options_popup = True
            if imgui.is_item_hovered():
                wrapped_tooltip("GPU adapter, debug logging, and other settings")
            imgui.same_line()
            if imgui.button(f"{fa.ICON_FA_XMARK}  Quit", imgui.ImVec2(quit_w, btn_h)) or imgui.is_key_pressed(imgui.Key.escape):
                self.selected_path = None
                hello_imgui.get_runner_params().app_shall_exit = True
            pop_button_style()

            imgui.pop_id()

        imgui.pop_style_var(4)
        imgui.pop_style_color(8)


if __name__ == "__main__":
    pass
