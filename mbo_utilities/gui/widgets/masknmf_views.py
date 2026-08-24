"""masknmf view selector: switch the displayed movie between demixing components."""

import threading
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui.widgets._base import Widget
from mbo_utilities.gui._imgui_helpers import set_tooltip


def _results_file(fpath) -> Path | None:
    if not fpath:
        return None
    if isinstance(fpath, (list, tuple)):
        fpath = fpath[0]
    p = Path(fpath)
    d = p if p.is_dir() else p.parent
    f = d / "demixing_results.hdf5"
    return f if f.exists() else None


class MaskNMFViewsWidget(Widget):
    """Combo bar to display masknmf demixing components (signal, residual,
    colorful ACs, background, PMD, trend) in place of the registered movie."""

    name = "MaskNMF View"
    priority = 5

    _LABELS = ("Signal (AC)", "Colorful", "Residual", "Background", "PMD", "Trend")

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._file = _results_file(getattr(parent, "fpath", None))
        self._views = None
        self._loading = False
        self._error = None
        self._pending = None
        self._current = "Registered"
        self._orig = None

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        return _results_file(getattr(parent, "fpath", None)) is not None

    def _load_async(self):
        def _run():
            try:
                from mbo_utilities.masknmf.views import load_demix_views

                self._views = load_demix_views(self._file)
            except Exception as e:
                self._error = str(e)
            finally:
                self._loading = False

        self._loading = True
        threading.Thread(target=_run, daemon=True, name="masknmf-views").start()

    def _apply(self, label: str):
        iw = self.parent.image_widget
        if self._orig is None:
            self._orig = iw.data[0]
        target = self._orig if label == "Registered" else (self._views or {}).get(label)
        if target is None:
            return
        try:
            iw.data[0] = target
        except Exception as e:
            self._error = str(e)
            self.parent.logger.exception(f"View switch failed: {e}")
            return
        self._current = label
        # the swap clears window funcs / spatial funcs; re-apply panel state
        self.parent._update_window_funcs()
        self.parent._rebuild_spatial_func()
        self.parent.logger.info(f"View: {label}")

    def draw(self) -> None:
        parent = self.parent

        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(0.8, 0.8, 0.2, 1.0), "MaskNMF View")
        imgui.spacing()

        if self._views is not None:
            labels = ["Registered"] + list(self._views)
        else:
            labels = ["Registered"] + list(self._LABELS)
        try:
            idx = labels.index(self._current)
        except ValueError:
            idx = 0

        imgui.set_next_item_width(imgui.get_content_region_avail().x * 0.55)
        changed, new_idx = imgui.combo("Component", idx, labels)
        set_tooltip(
            "Demixing component to display: registered movie, denoised signal "
            "(a·c), per-ROI colored signal, residual, fluctuating background, "
            "PMD reconstruction, or detrend baseline."
        )
        if changed and labels[new_idx] != self._current:
            sel = labels[new_idx]
            if sel == "Registered" or self._views is not None:
                self._apply(sel)
            else:
                self._pending = sel
                if not self._loading and self._error is None:
                    self._load_async()

        if self._loading:
            imgui.text_disabled("Loading demixing results...")
        elif self._error:
            imgui.text_colored(imgui.ImVec4(1.0, 0.4, 0.4, 1.0), "Load failed")
            set_tooltip(self._error)

        if self._views is not None and self._pending:
            sel, self._pending = self._pending, None
            self._apply(sel)

    def cleanup(self) -> None:
        if self._orig is not None and self._current != "Registered":
            try:
                self.parent.image_widget.data[0] = self._orig
            except Exception:
                pass
