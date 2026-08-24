"""BioHPC / Lab4 tab — proof-of-concept no-code cluster workflow.

A self-contained tab that gates a Lab4 data workflow behind a demo login
(``loson`` / ``treadmill``), then exposes four sub-tabs that mirror what a
scientist does without writing code:

* **Transfer** — pick a BioHPC folder and "send" the loaded dataset there.
* **Metadata** — set the Lab4 recording identity (the DataJoint key) and edit
  the acquisition metadata that travels with the data.
* **Analysis** — choose an extraction engine + analyses, tune parameters, and
  watch the ``processing_config`` build live, then dispatch to BioHPC.
* **Jobs** — a simulated SLURM queue with per-job progress and logs.

Everything is simulated locally — no data leaves the machine, no cluster is
contacted — so this is a UI/UX proof of concept, not a live client. Styling
mirrors the Suite2p settings panel and the launch file dialog.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from imgui_bundle import (
    imgui,
    imgui_ctx,
    hello_imgui,
    icons_fontawesome_6 as fa,
)

from mbo_utilities.gui._imgui_helpers import set_tooltip

__all__ = ["draw_biohpc_tab"]

# demo credentials for the proof of concept. real auth never belongs in
# source — this gate only unlocks a local, simulated workflow.
_DEMO_USER = "loson"
_DEMO_PASS = "treadmill"

# palette — mirrors file_dialog.py (accent/card) and settings.py (headers).
_COL_ACCENT = imgui.ImVec4(0.20, 0.50, 0.85, 1.0)
_COL_ACCENT_HOVER = imgui.ImVec4(0.25, 0.55, 0.90, 1.0)
_COL_ACCENT_ACTIVE = imgui.ImVec4(0.15, 0.45, 0.80, 1.0)
_COL_VIOLET = imgui.ImVec4(0.42, 0.36, 0.90, 1.0)  # lab4 brand accent
_COL_CARD = imgui.ImVec4(0.16, 0.16, 0.17, 1.0)
_COL_TITLE = imgui.ImVec4(1.0, 0.85, 0.4, 1.0)   # orange section header
_COL_SUB = imgui.ImVec4(0.55, 0.75, 1.0, 1.0)    # blue subsection header
_COL_DIM = imgui.ImVec4(0.6, 0.6, 0.6, 1.0)
_COL_OK = imgui.ImVec4(0.4, 1.0, 0.4, 1.0)
_COL_WARN = imgui.ImVec4(1.0, 0.75, 0.3, 1.0)
_COL_ERR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)
_COL_PATH = imgui.ImVec4(0.55, 0.85, 0.55, 1.0)  # resolved destination path
_COL_CODE = imgui.ImVec4(0.75, 0.85, 0.65, 1.0)

# preset BioHPC destinations (label, description). the last entry switches
# on a free-form custom path field.
_CUSTOM_LABEL = "Custom path..."
_DESTINATIONS: list[tuple[str, str]] = [
    ("/archive/mbo/raw", "Raw acquisitions - long-term archive"),
    ("/archive/mbo/assembled", "Assembled volumes (zarr / tiff)"),
    ("/scratch/mbo/suite2p", "Suite2p outputs - fast scratch"),
    ("/work/mbo/shared", "Shared lab working set"),
    (_CUSTOM_LABEL, "Enter a destination path manually"),
]

# simulated transfer duration and its labeled phases (fraction upper bound).
_XFER_SECONDS = 2.6
_XFER_PHASES: list[tuple[float, str]] = [
    (0.15, "Connecting to biohpc..."),
    (0.30, "Authenticating..."),
    (0.45, "Preparing payload..."),
    (0.95, "Transferring..."),
    (1.00, "Verifying..."),
]

# ---- Lab4 domain fixtures ------------------------------------------------
# Values a scientist picks from instead of typing free-form. These mirror the
# Lab4 DataJoint pipeline (Experimenter -> Subject -> Session -> Recording ->
# Params -> Extract -> [engine] -> Dfof) and its core/analysis library.

_EXPERIMENTERS = ["loson", "jdoe", "asmith", "mkim", "rpatel"]
_DEVICES = [
    "AOD-2P", "Bruker-2P", "Mini2P", "SLM-2P",
    "BehaviorMate", "DeepLabCut", "SLEAP", "Doric-1P", "LFP",
]
_INDICATORS = ["GCaMP6f", "GCaMP6s", "GCaMP7f", "GCaMP8f", "GCaMP8m",
               "GCaMP8s", "jRGECO1a"]
_REGIONS = ["CA1", "CA3", "DG", "V1", "mPFC", "RSC", "S1"]

# extraction engines (label, config key, description).
_ENGINES: list[tuple[str, str, str]] = [
    ("Suite2p", "suite2p", "ROI detection + deconvolution (default)"),
    ("Caiman", "caiman", "CNMF source extraction"),
    ("Kasper", "kasper", "Dendritic spine extraction"),
    ("MaskNMF", "masknmf", "Demixing for dense labeling"),
]

# post-extraction analyses (label, config key, description). Each becomes a
# checkbox; selected keys land in processing_config["analysis"].
_ANALYSES: list[tuple[str, str, str]] = [
    ("Place fields", "place_fields", "Spatial firing maps along the track"),
    ("Spatial information", "spatial_info", "Skaggs bits/spike + shuffle test"),
    ("Tuning curves", "tuning", "Activity vs. position / speed"),
    ("Ripple detection", "ripples", "SWR events from LFP band"),
    ("Bayesian decoding", "decoding", "Reconstruct position from activity"),
    ("Speed tuning", "speed_tuning", "Firing rate vs. running speed"),
]


def _icon(name: str) -> str:
    """FontAwesome glyph by constant name, or "" if this build lacks it."""
    return getattr(fa, name, "")


def _center_text(text: str, color: "imgui.ImVec4 | None" = None) -> None:
    avail_w = imgui.get_content_region_avail().x
    tw = imgui.calc_text_size(text).x
    if tw >= avail_w:
        _wrapped(text, color)
        return
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail_w - tw) * 0.5)
    if color is None:
        imgui.text(text)
    else:
        imgui.text_colored(color, text)


def _field_row(label: str, label_w: float = 6.0) -> None:
    """Dim label on the left of a same-line input; stacks when too narrow.

    Caller follows with ``imgui.set_next_item_width(-1)`` so the input fills
    the remaining width and never clips off the right edge of a narrow panel.
    """
    imgui.align_text_to_frame_padding()
    imgui.text_colored(_COL_DIM, label)
    col_x = max(
        hello_imgui.em_size(label_w),
        imgui.calc_text_size(label).x + imgui.get_style().item_spacing.x * 2,
    )
    if imgui.get_content_region_avail().x - col_x >= hello_imgui.em_size(8):
        imgui.same_line(col_x)


def _wrapped(text: str, color: "imgui.ImVec4 | None" = None) -> None:
    """Text that wraps at the panel's visible right edge instead of clipping.

    Explicit wrap pos, not 0.0: in a horizontal-scrollbar child 0.0 resolves
    to the scrollable content edge, and over-wide wrapped lines then keep the
    content permanently inflated.
    """
    imgui.push_text_wrap_pos(
        imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
    )
    if color is None:
        imgui.text(text)
    else:
        imgui.text_colored(color, text)
    imgui.pop_text_wrap_pos()


def _desc(text: str) -> None:
    """Indented, wrapped dim helper line beneath a control."""
    imgui.indent(hello_imgui.em_size(0.5))
    _wrapped(text, _COL_DIM)
    imgui.unindent(hello_imgui.em_size(0.5))


@contextmanager
def _primary_style():
    """Accent-filled button — the one action worth clicking on the panel."""
    imgui.push_style_color(imgui.Col_.button, _COL_ACCENT)
    imgui.push_style_color(imgui.Col_.button_hovered, _COL_ACCENT_HOVER)
    imgui.push_style_color(imgui.Col_.button_active, _COL_ACCENT_ACTIVE)
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 1.0, 1.0, 1.0))
    imgui.push_style_var(imgui.StyleVar_.frame_rounding, 6.0)
    try:
        yield
    finally:
        imgui.pop_style_var(1)
        imgui.pop_style_color(4)


@contextmanager
def _ghost_style():
    """Faint surface-tint button for secondary actions (sign out)."""
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(1, 1, 1, 0.05))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1, 1, 1, 0.10))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(1, 1, 1, 0.16))
    try:
        yield
    finally:
        imgui.pop_style_color(3)


def _phase_for(frac: float) -> str:
    for upper, label in _XFER_PHASES:
        if frac <= upper:
            return label
    return _XFER_PHASES[-1][1]


def _human_bytes(n: "int | None") -> str:
    if not n:
        return ""
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if v < 1024.0 or unit == "PB":
            return f"{v:.0f} {unit}" if unit == "B" else f"{v:.1f} {unit}"
        v /= 1024.0
    return f"{v:.1f} PB"


# ---- simulated Lab4 job (SLURM row on biohpc) ----------------------------

_JOB_SECONDS = 7.0
_JOB_PHASES: list[tuple[float, str]] = [
    (0.10, "Queued on biohpc..."),
    (0.25, "Staging inputs..."),
    (0.45, "Extraction..."),
    (0.75, "Computing analyses..."),
    (0.95, "Writing results..."),
    (1.00, "Finalizing..."),
]


class _Job:
    """One simulated Lab4 compute row dispatched to BioHPC."""

    def __init__(self, jid: int, slurm: int, recording: str, engine: str,
                 analyses: list[str], target: str, start: float):
        self.jid = jid
        self.slurm = slurm
        self.recording = recording
        self.engine = engine
        self.analyses = analyses
        self.target = target
        self.start = start
        self.log_open = False

    def frac(self) -> float:
        if self.target != "biohpc":
            return 0.0  # manual: never auto-runs
        f = (time.time() - self.start) / _JOB_SECONDS
        return 0.0 if f < 0.0 else (1.0 if f > 1.0 else f)

    def status(self) -> str:
        if self.target != "biohpc":
            return "manual"
        f = self.frac()
        if f >= 1.0:
            return "succeeded"
        if f <= 0.10:  # matches the first _JOB_PHASES boundary
            return "queued"
        return "running"

    def phase(self) -> str:
        f = self.frac()
        for upper, label in _JOB_PHASES:
            if f <= upper:
                return label
        return _JOB_PHASES[-1][1]

    def log_lines(self) -> list[str]:
        st = self.status()
        head = f"slurm {self.slurm}  {self.engine}  {self.recording}"
        if st == "manual":
            return [head, "target=manual — run locally, not dispatched"]
        lines = [head]
        seen = set()
        cutoff = self.frac()
        for upper, label in _JOB_PHASES:
            if cutoff >= min(upper, 0.999) or st == "succeeded":
                if label not in seen:
                    lines.append(f"  [{int(upper * 100):3d}%] {label}")
                    seen.add(label)
        if st == "succeeded":
            lines.append("  done — results at /work/mbo/results")
        return lines


class _BioHpcPanel:
    """Per-parent controller: login, transfer, metadata, analysis, jobs."""

    def __init__(self, parent: Any):
        self.parent = parent
        self.authenticated = False
        self.username = ""
        self.password = ""
        self.error = ""

        # transfer
        self.dest_index = 0
        self.custom_dest = ""
        self.subfolder = ""
        self.opt_compress = True
        self.opt_verify = True
        self.opt_overwrite = False
        self.opt_metadata = True
        self.xfer_active = False
        self.xfer_done = False
        self.xfer_start = 0.0
        self.xfer_dest = ""

        # lab4 recording identity (the DataJoint key)
        self.experimenter = 0
        self.expt_group = "hpc-place-cells"
        self.subject = "M231"
        self.session = "2026-08-21"
        self.recording_idx = 1
        self.device = 1
        self.indicator = 3
        self.region = 0

        # analysis config
        self.engine = 0
        self.opt_dfof = True
        self.opt_neuropil = True
        self.opt_deconv = False
        self.opt_behaviormate = True
        self.analyses = {key: (key in ("place_fields", "spatial_info", "tuning"))
                         for _, key, _ in _ANALYSES}
        self.pf_bins = 100
        self.pf_min_rate = 1.0
        self.compute_target = 0  # 0=biohpc, 1=manual
        self.planes_per_gpu = 4

        # job queue
        self.jobs: list[_Job] = []
        self._job_counter = 0
        self._slurm_counter = 4180231

    # -- auth -----------------------------------------------------------

    def _attempt_login(self) -> None:
        if self.username.strip() == _DEMO_USER and self.password == _DEMO_PASS:
            self.authenticated = True
            self.error = ""
            self.password = ""
            self.experimenter = _EXPERIMENTERS.index(_DEMO_USER)
            self._log(f"biohpc: signed in as {_DEMO_USER}")
        else:
            self.error = "Invalid credentials."
            self.password = ""

    def _sign_out(self) -> None:
        self.authenticated = False
        self.password = ""
        self.error = ""
        self.xfer_active = False
        self.xfer_done = False

    def _log(self, msg: str) -> None:
        try:
            self.parent.logger.info(msg)
        except Exception:
            pass

    # -- source / destination resolution --------------------------------

    def _source_paths(self) -> list[Path]:
        fpath = getattr(self.parent, "fpath", None)
        try:
            if isinstance(fpath, (list, tuple)):
                return [Path(str(p)) for p in fpath if p]
            if fpath:
                return [Path(str(fpath))]
        except Exception:
            pass
        return []

    def _dataset_name(self, paths: list[Path]) -> str:
        if not paths:
            return ""
        if len(paths) == 1:
            return paths[0].name
        return paths[0].parent.name or "dataset"

    def _array_stats(self) -> "tuple[tuple[int, ...] | None, int | None]":
        try:
            arr = self.parent.image_widget.data[0]
            shape = tuple(int(s) for s in arr.shape)
            import numpy as np

            # math.prod keeps Python-int precision; np.prod would wrap int32
            # on Windows and report a garbage size for large movies.
            nbytes = math.prod(shape) * int(np.dtype(arr.dtype).itemsize)
            return shape, nbytes
        except Exception:
            return None, None

    def _base_dest(self) -> str:
        label = _DESTINATIONS[self.dest_index][0]
        if label == _CUSTOM_LABEL:
            return self.custom_dest.strip()
        return label

    def _dest_ready(self) -> bool:
        # a slash-only path normalizes to empty; require real folder content
        return bool(self._base_dest().strip("/"))

    def _resolved_dest(self, paths: list[Path]) -> str:
        base = self._base_dest()
        if not base.strip("/"):
            return ""
        parts = [base.rstrip("/")]
        sub = self.subfolder.strip().strip("/")
        if sub:
            parts.append(sub)
        name = self._dataset_name(paths)
        if name:
            parts.append(name)
        return f"{_DEMO_USER}@biohpc:" + "/".join(p for p in parts if p)

    # -- lab4 identity helpers ------------------------------------------

    def _recording_label(self) -> str:
        return f"{self.subject}_{self.session}_rec{self.recording_idx:02d}"

    def _datajoint_key(self) -> dict:
        return {
            "experimenter": _EXPERIMENTERS[self.experimenter],
            "expt_group": self.expt_group.strip(),
            "subject_id": self.subject.strip(),
            "session_date": self.session.strip(),
            "recording_idx": int(self.recording_idx),
        }

    def _h5_path(self) -> str:
        k = self._datajoint_key()
        return (f"/work/lab4/data/{k['experimenter']}/{k['subject_id']}/"
                f"{k['session_date']}/{self._recording_label()}.h5")

    def _fs(self) -> "float | None":
        try:
            v = getattr(self.parent.image_widget.data[0], "fs", None)
            return float(v) if v else None
        except Exception:
            return None

    def _nz(self) -> "int | None":
        try:
            return int(getattr(self.parent, "nz", 0)) or None
        except Exception:
            return None

    def _processing_config(self) -> dict:
        _, engine_key, _ = _ENGINES[self.engine]
        analysis: dict = {}
        for _, key, _ in _ANALYSES:
            if not self.analyses.get(key):
                continue
            if key == "place_fields":
                analysis[key] = {
                    "bins": int(self.pf_bins),
                    "min_peak_rate": round(float(self.pf_min_rate), 3),
                }
            else:
                analysis[key] = True
        cfg = {
            "recording": self._recording_label(),
            "extract": {
                "engine": engine_key,
                "dfof": bool(self.opt_dfof),
                "neuropil": bool(self.opt_neuropil),
                "deconvolve": bool(self.opt_deconv),
            },
            "behaviormate": {"sync": bool(self.opt_behaviormate)},
            "analysis": analysis,
            "compute": {
                "target": "biohpc" if self.compute_target == 0 else "manual",
                "planes_per_gpu": int(self.planes_per_gpu),
            },
        }
        fs = self._fs()
        if fs:
            cfg["extract"]["fs"] = round(fs, 3)
        nz = self._nz()
        if nz:
            cfg["extract"]["num_planes"] = nz
        return cfg

    # ==================================================================
    # draw: login
    # ==================================================================

    def draw_login(self) -> None:
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))

        avail_w = imgui.get_content_region_avail().x
        card_w = min(avail_w, hello_imgui.em_size(22))
        imgui.set_cursor_pos_x(
            imgui.get_cursor_pos_x() + max(0.0, (avail_w - card_w) * 0.5)
        )

        imgui.push_style_color(imgui.Col_.child_bg, _COL_CARD)
        imgui.push_style_var(imgui.StyleVar_.child_rounding, 8.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, hello_imgui.em_to_vec2(1.0, 0.8))
        child_flags = imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y
        try:
            with imgui_ctx.begin_child(
                "##biohpc_login_card",
                size=imgui.ImVec2(card_w, 0),
                child_flags=child_flags,
                window_flags=imgui.WindowFlags_.no_scrollbar,
            ):
                self._draw_login_body()
        finally:
            imgui.pop_style_var(2)
            imgui.pop_style_color(1)

    def _draw_login_body(self) -> None:
        _center_text(f"{_icon('ICON_FA_LOCK')}  Lab4 · BioHPC Login", _COL_ACCENT)
        _center_text("Sign in to move data and launch analysis", _COL_DIM)

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))
        imgui.separator()
        imgui.dummy(hello_imgui.em_to_vec2(0, 0.3))

        full_w = imgui.get_content_region_avail().x

        imgui.text_colored(_COL_DIM, f"{_icon('ICON_FA_USER')}  Username")
        imgui.set_next_item_width(full_w)
        _, self.username = imgui.input_text_with_hint(
            "##biohpc_user", "username", self.username
        )

        imgui.dummy(imgui.ImVec2(0, 2))
        imgui.text_colored(_COL_DIM, f"{_icon('ICON_FA_LOCK')}  Password")
        imgui.set_next_item_width(full_w)
        pflags = (
            imgui.InputTextFlags_.password | imgui.InputTextFlags_.enter_returns_true
        )
        enter, self.password = imgui.input_text_with_hint(
            "##biohpc_pass", "password", self.password, flags=pflags
        )

        if self.error:
            imgui.dummy(imgui.ImVec2(0, 2))
            _wrapped(
                f"{_icon('ICON_FA_CIRCLE_EXCLAMATION')}  {self.error}", _COL_ERR
            )

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.4))
        ready = bool(self.username.strip()) and bool(self.password)
        imgui.begin_disabled(not ready)
        with _primary_style():
            clicked = imgui.button(
                "Sign In", imgui.ImVec2(full_w, hello_imgui.em_size(1.9))
            )
        imgui.end_disabled()
        if clicked or (enter and ready):
            self._attempt_login()

        imgui.dummy(hello_imgui.em_to_vec2(0, 0.2))
        _center_text(f"Demo login:  {_DEMO_USER}  /  {_DEMO_PASS}", _COL_DIM)

    # ==================================================================
    # draw: authenticated shell + sub-tabs
    # ==================================================================

    def draw_authenticated(self) -> None:
        imgui.spacing()
        imgui.text_colored(
            _COL_TITLE, f"{_icon('ICON_FA_FLASK')}  Lab4 Workbench"
        )
        signout = f"{_icon('ICON_FA_RIGHT_FROM_BRACKET')}  Sign out"
        style = imgui.get_style()
        so_w = imgui.calc_text_size(signout).x + style.frame_padding.x * 2
        imgui.same_line()
        avail = imgui.get_content_region_avail().x
        if avail < so_w + hello_imgui.em_size(0.5):
            imgui.new_line()
            avail = imgui.get_content_region_avail().x
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + max(0.0, avail - so_w))
        with _ghost_style():
            signed_out = imgui.small_button(signout)
        if signed_out:
            self._sign_out()
            return

        _wrapped(
            f"Signed in as {_DEMO_USER}",
            _COL_DIM,
        )
        imgui.spacing()

        if imgui.begin_tab_bar("##biohpc_subtabs"):
            if imgui.begin_tab_item(f"{_icon('ICON_FA_CLOUD_ARROW_UP')} Transfer")[0]:
                imgui.spacing()
                self._draw_transfer_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item(f"{_icon('ICON_FA_TAGS')} Metadata")[0]:
                imgui.spacing()
                self._draw_metadata_tab()
                imgui.end_tab_item()
            if imgui.begin_tab_item(f"{_icon('ICON_FA_SLIDERS')} Analysis")[0]:
                imgui.spacing()
                self._draw_analysis_tab()
                imgui.end_tab_item()
            njobs = len(self.jobs)
            jobs_label = f"{_icon('ICON_FA_LIST_CHECK')} Jobs"
            if njobs:
                jobs_label += f" ({njobs})"
            if imgui.begin_tab_item(jobs_label)[0]:
                imgui.spacing()
                self._draw_jobs_tab()
                imgui.end_tab_item()
            imgui.end_tab_bar()

    # ==================================================================
    # sub-tab: Transfer
    # ==================================================================

    def _draw_transfer_tab(self) -> None:
        paths = self._source_paths()
        self._draw_source(paths)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        self._draw_destination(paths)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        self._draw_options()
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        self._draw_send(paths)

    def _draw_source(self, paths: list[Path]) -> None:
        imgui.text_colored(_COL_SUB, "Source")
        imgui.dummy(imgui.ImVec2(0, 2))
        if not paths:
            _wrapped("No dataset loaded.", _COL_DIM)
            return

        name = paths[0].name if len(paths) == 1 else f"{len(paths)} files"
        _field_row("Dataset")
        _wrapped(name)

        shape, nbytes = self._array_stats()
        if shape is not None:
            _field_row("Size")
            size_txt = str(shape)
            human = _human_bytes(nbytes)
            if human:
                size_txt += f"  ({human})"
            _wrapped(size_txt)

        imgui.text_colored(_COL_DIM, "Path")
        imgui.indent(hello_imgui.em_size(0.5))
        _wrapped(str(paths[0].parent), _COL_DIM)
        imgui.unindent(hello_imgui.em_size(0.5))

    def _draw_destination(self, paths: list[Path]) -> None:
        imgui.text_colored(_COL_SUB, "Destination")
        imgui.dummy(imgui.ImVec2(0, 2))

        labels = [d[0] for d in _DESTINATIONS]
        _field_row("Folder")
        imgui.set_next_item_width(-1)
        changed, self.dest_index = imgui.combo("##biohpc_dest", self.dest_index, labels)
        if changed:
            self.xfer_done = False

        _desc(_DESTINATIONS[self.dest_index][1])

        if labels[self.dest_index] == _CUSTOM_LABEL:
            _field_row("Path")
            imgui.set_next_item_width(-1)
            ch, self.custom_dest = imgui.input_text_with_hint(
                "##biohpc_custom", "/path/on/biohpc", self.custom_dest
            )
            if ch:
                self.xfer_done = False

        _field_row("Subfolder")
        imgui.set_next_item_width(-1)
        ch, self.subfolder = imgui.input_text_with_hint(
            "##biohpc_sub", "optional", self.subfolder
        )
        if ch:
            self.xfer_done = False

        dest = self._resolved_dest(paths)
        imgui.dummy(imgui.ImVec2(0, 3))
        if dest:
            imgui.text_colored(_COL_DIM, "Uploads to")
            imgui.indent(hello_imgui.em_size(0.5))
            _wrapped(dest, _COL_PATH)
            imgui.unindent(hello_imgui.em_size(0.5))
        else:
            _wrapped("Enter a destination path.", _COL_DIM)

    def _draw_options(self) -> None:
        imgui.text_colored(_COL_SUB, "Options")
        imgui.dummy(imgui.ImVec2(0, 2))
        _, self.opt_compress = imgui.checkbox(
            "Compress before upload", self.opt_compress
        )
        set_tooltip("Pack the dataset before sending to reduce transfer size.")
        _, self.opt_verify = imgui.checkbox("Verify checksum", self.opt_verify)
        set_tooltip("Re-hash after upload to confirm an intact copy.")
        _, self.opt_overwrite = imgui.checkbox("Overwrite existing", self.opt_overwrite)
        set_tooltip("Replace files already present at the destination.")
        _, self.opt_metadata = imgui.checkbox(
            "Include metadata sidecar", self.opt_metadata
        )
        set_tooltip("Write a .json of acquisition metadata alongside the data.")

    def _draw_send(self, paths: list[Path]) -> None:
        dest = self._resolved_dest(paths)
        can_send = bool(paths) and self._dest_ready() and not self.xfer_active

        avail_w = imgui.get_content_region_avail().x
        imgui.begin_disabled(not can_send)
        with _primary_style():
            send = imgui.button(
                f"{_icon('ICON_FA_CLOUD_ARROW_UP')}  Send to BioHPC",
                imgui.ImVec2(avail_w, hello_imgui.em_size(2.1)),
            )
        imgui.end_disabled()
        if send:
            self._start_transfer(dest)

        if not paths:
            _wrapped("Load a dataset to enable transfer.", _COL_DIM)
        elif not self._dest_ready():
            _wrapped("Enter a destination path.", _COL_DIM)

        if self.xfer_active:
            self._draw_progress()
        elif self.xfer_done:
            imgui.spacing()
            _wrapped(
                f"{_icon('ICON_FA_CIRCLE_CHECK')}  Transfer complete", _COL_OK
            )
            imgui.text_colored(_COL_DIM, "Sent to")
            imgui.same_line()
            _wrapped(self.xfer_dest, _COL_PATH)

    def _draw_progress(self) -> None:
        frac = (time.time() - self.xfer_start) / _XFER_SECONDS
        if frac >= 1.0:
            frac = 1.0
            self.xfer_active = False
            self.xfer_done = True
            self._log(f"biohpc: simulated transfer complete -> {self.xfer_dest}")

        imgui.spacing()
        avail_w = imgui.get_content_region_avail().x
        imgui.progress_bar(
            frac,
            imgui.ImVec2(avail_w, hello_imgui.em_size(1.3)),
            f"{int(frac * 100)}%",
        )
        imgui.text_colored(_COL_ACCENT, _phase_for(frac))

    def _start_transfer(self, dest: str) -> None:
        self.xfer_active = True
        self.xfer_done = False
        self.xfer_start = time.time()
        self.xfer_dest = dest
        name = self._dataset_name(self._source_paths())
        self._log(f"biohpc: simulated upload {name} -> {dest}")

    # ==================================================================
    # sub-tab: Metadata (Lab4 recording identity + array metadata)
    # ==================================================================

    def _draw_metadata_tab(self) -> None:
        imgui.text_colored(_COL_SUB, "Recording identity")
        _wrapped("The Lab4 DataJoint key this data registers under.", _COL_DIM)
        imgui.dummy(imgui.ImVec2(0, 2))

        _field_row("Experimenter", 8)
        imgui.set_next_item_width(-1)
        _, self.experimenter = imgui.combo(
            "##l4_exp", self.experimenter, _EXPERIMENTERS
        )

        _field_row("Experiment group", 8)
        imgui.set_next_item_width(-1)
        _, self.expt_group = imgui.input_text_with_hint(
            "##l4_group", "study / cohort", self.expt_group
        )

        _field_row("Subject", 8)
        imgui.set_next_item_width(-1)
        _, self.subject = imgui.input_text_with_hint(
            "##l4_subj", "mouse id", self.subject
        )

        _field_row("Session date", 8)
        imgui.set_next_item_width(-1)
        _, self.session = imgui.input_text_with_hint(
            "##l4_sess", "YYYY-MM-DD", self.session
        )

        _field_row("Recording #", 8)
        imgui.set_next_item_width(-1)
        _, self.recording_idx = imgui.input_int("##l4_rec", self.recording_idx)
        if self.recording_idx < 1:
            self.recording_idx = 1

        _field_row("Device", 8)
        imgui.set_next_item_width(-1)
        _, self.device = imgui.combo("##l4_dev", self.device, _DEVICES)

        _field_row("Indicator", 8)
        imgui.set_next_item_width(-1)
        _, self.indicator = imgui.combo("##l4_ind", self.indicator, _INDICATORS)

        _field_row("Region", 8)
        imgui.set_next_item_width(-1)
        _, self.region = imgui.combo("##l4_reg", self.region, _REGIONS)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored(_COL_SUB, "Resolves to")
        imgui.dummy(imgui.ImVec2(0, 2))
        self._kv("Recording", self._recording_label(), _COL_PATH)
        self._kv("Device", f"{_DEVICES[self.device]}  ·  "
                 f"{_INDICATORS[self.indicator]}  ·  {_REGIONS[self.region]}")
        imgui.text_colored(_COL_DIM, "Standardized H5")
        imgui.indent(hello_imgui.em_size(0.5))
        _wrapped(self._h5_path(), _COL_PATH)
        imgui.unindent(hello_imgui.em_size(0.5))

        imgui.spacing()
        with _primary_style():
            if imgui.button(
                f"{_icon('ICON_FA_DATABASE')}  Register recording in Lab4",
                imgui.ImVec2(-1, hello_imgui.em_size(1.9)),
            ):
                key = self._datajoint_key()
                self._log(f"lab4: simulated Recording.insert1({key})")
                self._registered_key = key
        if getattr(self, "_registered_key", None) is not None:
            _wrapped(
                f"{_icon('ICON_FA_CIRCLE_CHECK')}  Registered "
                f"{self._recording_label()} (simulated)",
                _COL_OK,
            )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # real acquisition metadata editor, embedded under a collapsing header
        if imgui.collapsing_header("Acquisition metadata (from array)"):
            _wrapped(
                "Fields read from the loaded array; edits travel with the H5.",
                _COL_DIM,
            )
            imgui.dummy(imgui.ImVec2(0, 2))
            try:
                from mbo_utilities.gui._metadata_editor import (
                    draw_metadata_editor_content,
                )
                draw_metadata_editor_content(self.parent)
            except Exception as e:
                _wrapped(f"metadata editor unavailable: {e}", _COL_ERR)

    def _kv(self, label: str, value: str, color: "imgui.ImVec4 | None" = None) -> None:
        imgui.text_colored(_COL_DIM, label)
        imgui.same_line(hello_imgui.em_size(8))
        _wrapped(value, color)

    # ==================================================================
    # sub-tab: Analysis (choose engine + analyses, config, dispatch)
    # ==================================================================

    def _draw_analysis_tab(self) -> None:
        imgui.text_colored(_COL_SUB, "Extraction engine")
        imgui.dummy(imgui.ImVec2(0, 2))
        for i, (label, _, desc) in enumerate(_ENGINES):
            if imgui.radio_button(f"{label}##eng", self.engine == i):
                self.engine = i
            set_tooltip(desc)
            if self.engine == i:
                _desc(desc)

        imgui.dummy(imgui.ImVec2(0, 2))
        _, self.opt_dfof = imgui.checkbox("Compute dF/F", self.opt_dfof)
        set_tooltip("Baseline-normalized fluorescence (Dfof table).")
        _, self.opt_neuropil = imgui.checkbox("Neuropil correction", self.opt_neuropil)
        _, self.opt_deconv = imgui.checkbox("Deconvolve spikes", self.opt_deconv)
        _, self.opt_behaviormate = imgui.checkbox(
            "BehaviorMate sync", self.opt_behaviormate
        )
        set_tooltip("Align imaging to treadmill position / lap events.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored(_COL_SUB, "Analyses to run")
        imgui.dummy(imgui.ImVec2(0, 2))
        for label, key, desc in _ANALYSES:
            _, self.analyses[key] = imgui.checkbox(f"{label}##an", self.analyses[key])
            set_tooltip(desc)
            if self.analyses[key]:
                _desc(desc)

        if self.analyses.get("place_fields"):
            imgui.spacing()
            imgui.text_colored(_COL_DIM, "Place-field parameters")
            imgui.indent(hello_imgui.em_size(0.5))
            imgui.text_colored(_COL_DIM, "Track bins")
            imgui.set_next_item_width(-1)
            _, self.pf_bins = imgui.slider_int("##pf_bins", self.pf_bins, 20, 400)
            imgui.text_colored(_COL_DIM, "Min peak rate (Hz)")
            imgui.set_next_item_width(-1)
            _, self.pf_min_rate = imgui.input_float(
                "##pf_rate", self.pf_min_rate, 0.1, 1.0, "%.2f"
            )
            if self.pf_min_rate < 0.0:
                self.pf_min_rate = 0.0
            imgui.unindent(hello_imgui.em_size(0.5))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored(_COL_SUB, "Compute target")
        imgui.dummy(imgui.ImVec2(0, 2))
        if imgui.radio_button("BioHPC (SLURM)##ct", self.compute_target == 0):
            self.compute_target = 0
        if imgui.radio_button("Manual (run locally)##ct", self.compute_target == 1):
            self.compute_target = 1
        if self.compute_target == 0:
            imgui.text_colored(_COL_DIM, "Planes / GPU")
            imgui.set_next_item_width(-1)
            _, self.planes_per_gpu = imgui.slider_int(
                "##ppg", self.planes_per_gpu, 1, 16
            )
            set_tooltip("Pack factor F: z-planes per GPU shard in the array job.")

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        imgui.text_colored(_COL_SUB, "processing_config")
        _wrapped("Built live from your choices — what the cluster consumes.", _COL_DIM)
        imgui.dummy(imgui.ImVec2(0, 2))
        cfg_json = json.dumps(self._processing_config(), indent=2)
        n_lines = cfg_json.count("\n") + 1
        box_h = min(hello_imgui.em_size(16), hello_imgui.em_size(0.1) +
                    n_lines * imgui.get_text_line_height_with_spacing())
        imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0.10, 0.10, 0.12, 1.0))
        imgui.push_style_color(imgui.Col_.text, _COL_CODE)
        imgui.input_text_multiline(
            "##l4_cfg",
            cfg_json,
            imgui.ImVec2(-1, box_h),
            flags=imgui.InputTextFlags_.read_only,
        )
        imgui.pop_style_color(2)

        imgui.spacing()
        n_sel = sum(1 for v in self.analyses.values() if v)
        can_dispatch = n_sel > 0
        target = "BioHPC" if self.compute_target == 0 else "locally"
        imgui.begin_disabled(not can_dispatch)
        with _primary_style():
            if imgui.button(
                f"{_icon('ICON_FA_ROCKET')}  Dispatch to {target}",
                imgui.ImVec2(-1, hello_imgui.em_size(2.1)),
            ):
                self._dispatch_job()
        imgui.end_disabled()
        if not can_dispatch:
            _wrapped("Select at least one analysis.", _COL_DIM)
        else:
            _wrapped(
                f"Queues {self._recording_label()} · {_ENGINES[self.engine][0]} · "
                f"{n_sel} analysis{'es' if n_sel != 1 else ''}",
                _COL_DIM,
            )

    def _dispatch_job(self) -> None:
        self._job_counter += 1
        self._slurm_counter += 1
        target = "biohpc" if self.compute_target == 0 else "manual"
        selected = [key for _, key, _ in _ANALYSES if self.analyses.get(key)]
        job = _Job(
            jid=self._job_counter,
            slurm=self._slurm_counter,
            recording=self._recording_label(),
            engine=_ENGINES[self.engine][0],
            analyses=selected,
            target=target,
            start=time.time(),
        )
        self.jobs.insert(0, job)
        self._log(
            f"lab4: simulated dispatch {job.recording} "
            f"[{job.engine}] target={target} slurm={job.slurm}"
        )

    # ==================================================================
    # sub-tab: Jobs (simulated SLURM queue)
    # ==================================================================

    def _draw_jobs_tab(self) -> None:
        if not self.jobs:
            _wrapped("No jobs yet.", _COL_DIM)
            _wrapped("Dispatch one from the Analysis tab to see it here.", _COL_DIM)
            return

        active = sum(1 for j in self.jobs if j.status() in ("queued", "running"))
        # succeeded and manual are both terminal — 'Clear finished' removes both
        done = sum(1 for j in self.jobs if j.status() in ("succeeded", "manual"))
        _wrapped(
            f"{active} active · {done} finished · {len(self.jobs)} total", _COL_DIM
        )
        with _ghost_style():
            if imgui.small_button("Clear finished"):
                self.jobs = [j for j in self.jobs
                             if j.status() not in ("succeeded", "manual")]

        imgui.spacing()
        # stacked cards — never clip on a narrow panel the way a wide table would
        for job in self.jobs:
            self._draw_job_card(job)

    def _draw_job_card(self, job: _Job) -> None:
        st = job.status()
        caret = _icon("ICON_FA_CARET_DOWN") if job.log_open else _icon("ICON_FA_CARET_RIGHT")
        if imgui.small_button(f"{caret} #{job.jid}##job{job.jid}"):
            job.log_open = not job.log_open
        set_tooltip(f"slurm {job.slurm}")
        imgui.same_line()
        self._status_badge(st)

        imgui.indent(hello_imgui.em_size(0.5))
        _wrapped(job.recording, _COL_PATH)
        _wrapped(f"{job.engine}  ·  slurm {job.slurm}", _COL_DIM)

        if st in ("queued", "running"):
            imgui.progress_bar(
                job.frac(),
                imgui.ImVec2(-1, hello_imgui.em_size(1.1)),
                job.phase(),
            )
        elif st == "succeeded":
            imgui.text_colored(_COL_OK, f"{_icon('ICON_FA_CIRCLE_CHECK')}  done")
        else:
            _wrapped("not dispatched — run locally", _COL_DIM)

        if job.log_open:
            self._draw_job_log(job)

        imgui.unindent(hello_imgui.em_size(0.5))
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

    def _draw_job_log(self, job: _Job) -> None:
        imgui.push_style_color(imgui.Col_.child_bg, imgui.ImVec4(0.10, 0.10, 0.12, 1.0))
        with imgui_ctx.begin_child(
            f"##joblog{job.jid}",
            imgui.ImVec2(-1, hello_imgui.em_size(7)),
            imgui.ChildFlags_.borders,
        ):
            for line in job.log_lines():
                _wrapped(line, _COL_CODE)
        imgui.pop_style_color(1)

    def _status_badge(self, status: str) -> None:
        color, glyph = {
            "queued": (_COL_WARN, "ICON_FA_CLOCK"),
            "running": (_COL_ACCENT, "ICON_FA_SPINNER"),
            "succeeded": (_COL_OK, "ICON_FA_CIRCLE_CHECK"),
            "manual": (_COL_DIM, "ICON_FA_HAND"),
        }.get(status, (_COL_DIM, "ICON_FA_CIRCLE"))
        imgui.text_colored(color, f"{_icon(glyph)}  {status}")


def _panel(parent: Any) -> _BioHpcPanel:
    panel = getattr(parent, "_biohpc_panel", None)
    if panel is None:
        panel = _BioHpcPanel(parent)
        parent._biohpc_panel = panel
    return panel


def draw_biohpc_tab(parent: Any) -> None:
    """Render the BioHPC login gate, or the Lab4 workbench once signed in."""
    panel = _panel(parent)
    if panel.authenticated:
        panel.draw_authenticated()
    else:
        panel.draw_login()
