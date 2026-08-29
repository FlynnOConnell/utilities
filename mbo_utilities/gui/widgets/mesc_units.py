"""MESc unit selector: switch which MUnit of a ``.mesc`` file is displayed.

A ``.mesc`` holds one measurement unit per scan the operator ran — a z-stack,
a ribbon time series, a snapshot — and they are unrelated recordings with
different shapes. The launch picker chooses the first one to open; this widget
switches between them without leaving the viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from mbo_utilities.gui._imgui_helpers import set_tooltip
from mbo_utilities.gui.widgets._base import Widget

_ACCENT = imgui.ImVec4(0.8, 0.8, 0.2, 1.0)
_ERROR = imgui.ImVec4(1.0, 0.4, 0.4, 1.0)


def mesc_array_of(obj):
    """The `MescArray` behind a viewer array, or None.

    Peels the display wrappers (`_SqueezeSingletonDims`, `_ScrubTimingProxy`,
    …), each of which exposes its source as ``_arr``.
    """
    from mbo_utilities.arrays.mesc import MescArray

    for _ in range(8):
        if isinstance(obj, MescArray):
            return obj
        nxt = getattr(obj, "_arr", None)
        if nxt is None or nxt is obj:
            return None
        obj = nxt
    return None


def display_wrap(arr):
    """Wrap a reader the way the launch path does before handing it to the viewer."""
    from mbo_utilities.gui.run_gui import _ScrubTimingProxy, _SqueezeSingletonDims

    shape = getattr(arr, "shape", ())
    if len(shape) == 5 and any(shape[i] == 1 for i in range(3)):
        arr = _SqueezeSingletonDims(arr)
    return _ScrubTimingProxy(arr)


def _shape_text(shape) -> str:
    t, c, z, y, x = shape
    return f"{t}T x {c}C x {z}Z  ·  {y} x {x} px"


class MescUnitsWidget(Widget):
    """Combo bar to switch which MUnit of the open ``.mesc`` is displayed."""

    name = "MESc Units"
    priority = 4
    toggle_key = "preview.mesc_units"

    def __init__(self, parent: Any):
        super().__init__(parent)
        self._error: str | None = None

    @classmethod
    def is_supported(cls, parent: Any) -> bool:
        iw = getattr(parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return bool(len(data)) and mesc_array_of(data[0]) is not None

    # -- state -----------------------------------------------------------

    @property
    def _mesc(self):
        iw = getattr(self.parent, "image_widget", None)
        data = getattr(iw, "data", None) or []
        return mesc_array_of(data[0]) if len(data) else None

    def _cache(self) -> dict:
        """Open units, kept on the parent so a widget rebuild doesn't drop them."""
        cache = getattr(self.parent, "_mesc_unit_cache", None)
        if cache is None:
            cache = {}
            self.parent._mesc_unit_cache = cache
        return cache

    def _open_unit(self, path, key: str):
        """The `MescArray` for one unit, opening it on first use."""
        cache = self._cache()
        arr = cache.get(key)
        if arr is None:
            from mbo_utilities.arrays.mesc import MescArray

            arr = MescArray(path, unit=key)
            cache[key] = arr
        return arr

    # -- swapping --------------------------------------------------------

    def _install(self, arr) -> None:
        """Show `arr` in the viewer, re-deriving every per-dataset display state.

        Mirrors `mbo_utilities.gui._dialogs.load_new_data`: stale closures are
        dropped before the swap (the spatial func captured the previous unit's
        mean image and would be fed a differently shaped frame), then the
        widget re-derives its dimensions from the new array.
        """
        from mbo_utilities.gui._dialogs import _reset_per_data_state

        parent = self.parent
        iw = parent.image_widget

        _reset_per_data_state(parent)
        parent._rebuild_spatial_func()
        for proc in getattr(iw, "_image_processors", []) or []:
            proc.window_funcs = None
            proc.window_sizes = None
            proc.window_order = None

        display = display_wrap(arr)
        iw.data[0] = display
        # slider labels are a plain attribute on the widget; a unit swap can
        # change both the count and the meaning (Z-plane vs ROI), so they have
        # to be re-stamped alongside the data.
        iw._slider_dim_names = tuple(arr.slider_dim_labels) or None
        if getattr(iw, "n_sliders", 0) > 0:
            iw.indices = [0] * iw.n_sliders

        parent.shape = display.shape
        nt, nc, nz, _, _ = arr.shape
        parent.nc, parent.nz = nc, nz
        parent._custom_metadata = {}
        parent._update_window_funcs()
        parent.set_context_info()

        try:
            unit = arr.unit_key.rsplit("/", 1)[-1]
            iw.figure[0, 0].title = f"{Path(arr.filenames[0]).stem[:16]} · {unit}"
        except Exception:
            self.parent.logger.debug("subplot title update skipped", exc_info=True)

        # summary images / projections cache per-dataset statistics; the new
        # unit's are unrelated to the old one's.
        try:
            from mbo_utilities.gui.viewers import TimeSeriesViewer

            if isinstance(getattr(parent, "_viewer", None), TimeSeriesViewer):
                parent.refresh_zstats()
        except Exception:
            parent.logger.debug("zstats refresh skipped", exc_info=True)

        # widget support can differ between units (a snapshot has no time
        # axis, a z-stack no scan-phase). Rebinds parent._widgets to a new
        # list; the frame currently iterating the old one finishes safely.
        parent._refresh_widgets()
        parent.logger.info(f"MESc unit: {arr.unit_key}  shape={arr.shape}")

    def _switch(self, arr) -> None:
        self._error = None
        try:
            self._install(arr)
        except Exception as e:
            self._error = str(e)
            self.parent.logger.exception(f"MESc unit switch failed: {e}")

    # -- ui --------------------------------------------------------------

    def draw(self) -> None:
        mesc = self._mesc
        if mesc is None:
            return
        units = mesc.units
        # the unit opened at launch belongs in the cache too, so switching
        # away and back reuses it instead of opening the file a second time
        self._cache().setdefault(mesc.unit_key, mesc)

        imgui.spacing()
        imgui.text_colored(_ACCENT, "MESc Units")
        imgui.spacing()

        # `--roi 0` fans the ROIs of one unit across several subplots. Swapping
        # units there would replace only the first one and leave the rest
        # showing the old unit, so the selector stands down and says why.
        if len(self.parent.image_widget.data) > 1:
            imgui.text_disabled(f"{mesc.unit_key.rsplit('/', 1)[-1]} · split ROIs")
            imgui.text_disabled("Reopen without --roi to switch units.")
            return

        labels = [
            f"{u['munit']} · {u['modality_name']}"
            + (f" · {u['start_time'][:10]}" if u.get("start_time") else "")
            for u in units
        ]
        current = next(
            (i for i, u in enumerate(units) if u["key"] == mesc.unit_key), 0
        )

        imgui.set_next_item_width(imgui.get_content_region_avail().x * 0.9)
        changed, new_idx = imgui.combo("##mesc_unit", current, labels)
        set_tooltip(
            "Measurement unit to display. Each MUnit is one scan from this "
            "session — a z-stack, a time series, a snapshot — with its own "
            "shape and acquisition settings."
        )
        if changed and new_idx != current:
            unit = units[new_idx]
            try:
                arr = self._open_unit(mesc.filenames[0], unit["key"])
            except Exception as e:
                self._error = str(e)
                self.parent.logger.exception(f"cannot open {unit['key']}: {e}")
            else:
                self._switch(arr)
                return  # the array under us changed; redraw next frame

        info = units[current]
        imgui.text_disabled(_shape_text(info["shape"]))
        detail = f"{info['kind']} layout"
        fs = mesc.metadata.get("fs")
        if fs:
            detail += f"  ·  {fs:.1f} Hz"
        dur = info.get("duration_s")
        if dur:
            detail += f"  ·  {dur:.0f} s"
        if info["start_time"]:
            detail += f"  ·  {info['start_time'][:16].replace('T', ' ')}"
        imgui.text_disabled(detail)
        if info["comment"]:
            imgui.text_disabled(info["comment"][:48])
            set_tooltip(info["comment"])

        if self._error:
            imgui.text_colored(_ERROR, "Unit switch failed")
            set_tooltip(self._error)

    def cleanup(self) -> None:
        for arr in (getattr(self.parent, "_mesc_unit_cache", None) or {}).values():
            try:
                arr.close()
            except Exception:
                pass
        self.parent._mesc_unit_cache = None
