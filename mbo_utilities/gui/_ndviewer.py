"""MboNDViewer: fastplotlib NDWidget (ndwidget branch) behind the mbo
ImageWidget contract.

- ``.data`` — list-like of the ORIGINAL mbo arrays; ``data[i] = arr`` is a
  full swap (re-derives dims, rebuilds graphic/colorbar, resets funcs and
  indices).
- ``.indices`` — name-keyed, 0-based; iteration yields values in slider
  order. The reference space (what sliders display) is 1-based;
  ``_ref_to_index`` maps ref values to array indices.
- ``.graphics``/``.figure``/``.n_sliders``/``._sliders_ui``/
  ``._slider_dim_names``, window/spatial func routing, contrast resets.
- Dim names: caller ``slider_dim_names``, else the array's
  ``slider_dim_labels`` when they cover every slider axis, else canonical
  per-rank letters (5D TCZYX -> 't','c','z'). Spatial dims use reserved
  ``__row__``/``__col__``/``__rgb__``.
- Index changes fetch async on the rendercanvas loop. Offscreen canvases
  never register with the loop (cancelled fetches would wedge the serial
  path), so index setters fetch+render synchronously there.
"""

from __future__ import annotations

import contextlib
import math
from itertools import product
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6 as fa

from fastplotlib.utils import calculate_figure_shape
from fastplotlib.widgets.nd_widget import NDWidget, NDImage, NDImageProcessor
from fastplotlib.widgets.nd_widget._index import RangeContinuous
from fastplotlib.widgets.nd_widget._async import run_sync
from fastplotlib.widgets.nd_widget._ui import NDWidgetUI

__all__ = ["MboNDViewer", "MboNDImageProcessor", "_sample_array"]


# positional letters for slider axes 0/1/2 — the vendored widget's internal
# dim names, still accepted everywhere for name resolution ("t" is what
# preview_data/_apply_legacy_window_funcs and seed_fps callers pass)
_SLIDER_LETTERS = ("t", "z", "c")

# canonical per-rank axis letters for UNNAMED dims. mbo data is canonical
# (T, C, Z, Y, X) at 5D and (T, Z, Y, X) at 4D, so the letters must follow
# the rank — naming 5D axes with the vendored positional order t/z/c would
# put the 'z' slider on the C axis (real bug on unsqueezed MescArrays).
_CANONICAL_LETTERS = {1: ("t",), 2: ("t", "z"), 3: ("t", "c", "z")}


def _default_dim_letters(n: int) -> tuple[str, ...]:
    """canonical letters for ``n`` unnamed slider axes (+ dimN beyond 3)"""
    base = _CANONICAL_LETTERS.get(min(n, 3), ())
    return tuple(base) + tuple(f"dim{j}" for j in range(3, n))

# reserved spatial dim names — reserved so they can never collide with a
# user-supplied slider label
_ROW, _COL, _RGB = "__row__", "__col__", "__rgb__"


# How many positions to visit on each scrollable axis when sampling "the full
# data" for contrast/histograms, outermost axis first. Reading every frame is
# not an option — a MESc unit runs to 45k frames — so the range comes from an
# even sample. The counts multiply out to ~64 frames whatever the rank, and
# every read is a single 2D frame, so cost tracks frame size, not movie
# length.
VMINMAX_SAMPLE_COUNTS = {1: (64,), 2: (24, 3), 3: (12, 3, 2)}


def _sample_array(data) -> np.ndarray:
    """Values from a bounded, evenly spread sample of ``data``.

    2D data is returned whole. Anything deeper is sampled across *every*
    scrollable axis, not just the first — sampling time alone would report
    one channel's or one ROI's range as the whole array's.

    Each element of the sample is addressed with a fully scalar key so the
    read is one 2D frame; a partial key would pull a whole volume per sample
    (an isoview timepoint is every camera x every z-plane). This is the
    lazy-friendly IO pattern mbo readers are optimized for, unlike
    fastplotlib's ``subsample_array`` single strided multi-dim slice.
    """
    shape = tuple(getattr(data, "shape", ()) or ())
    if len(shape) <= 2:
        if not shape:
            return np.asarray(data)
        # read THROUGH the protocol: a protocol-only array (no __array__)
        # collapses to a useless 0-d object array under np.asarray(data)
        return np.asarray(data[:])
    scroll = shape[:-2]
    if any(n == 0 for n in scroll):
        return np.empty(0)
    counts = VMINMAX_SAMPLE_COUNTS.get(len(scroll), (8,) * len(scroll))
    grids = [
        np.unique(np.linspace(0, n - 1, min(int(n), c)).astype(int))
        for n, c in zip(scroll, counts)
    ]
    blocks = [
        np.asarray(data[tuple(int(i) for i in combo)]).ravel()
        for combo in product(*grids)
    ]
    return np.concatenate(blocks) if blocks else np.empty(0)


def _round_half_up(x) -> int:
    """Slider-dim transform: reference value -> array index, rounding .5 UP.

    fastplotlib's window indexer builds ``slice(f(t - s/2), f(t + s/2))``
    where ``f`` is this transform; the default identity uses banker's
    ``round()``, so an odd window size ``s`` covers ``s - 1`` or ``s + 1``
    frames depending on the parity of ``t - s//2`` (t-mean(5) covered 4
    frames). Half-up rounding makes the slice exactly
    ``[t - (s-1)/2, t + (s-1)/2 + 1)`` for every integer ``t`` — an odd
    window covers exactly ``s`` frames centered on ``t`` (the vendored
    contract) — while still mapping integer indices to themselves.
    """
    return int(math.floor(float(x) + 0.5))


def _ref_to_index(x) -> int:
    """1-based reference value -> 0-based array index (half-up)."""
    return _round_half_up(x) - 1


def _with_float32_cast(func):
    """Wrap a (possibly None) spatial func so its output is float32.

    Applied when window/spatial funcs run over INTEGER data: their float
    output would otherwise be cast straight into the graphic's integer
    texture (t-mean(5) displayed truncated ints).
    """
    if func is None:
        def _cast(a):
            return np.asarray(a, dtype=np.float32)
    else:
        def _cast(a):
            return np.asarray(func(a), dtype=np.float32)

    _cast._mbo_float_cast = True
    _cast._mbo_user_func = func
    return _cast


async def _noop_indices(*_args, **_kwargs):
    """Replacement ``_set_indices_`` for torn-down NDGraphics.

    An already-scheduled fetch coroutine may still run after the graphic and
    its reference dims are gone; this makes that a silent no-op instead of a
    KeyError/AttributeError inside the rendercanvas loop.
    """
    return None


class MboNDImageProcessor(NDImageProcessor):
    """``NDImageProcessor`` whose histogram uses mbo's scalar-key sampling.

    Upstream ``_recompute_histogram`` runs ``subsample_array`` — a single
    strided slice across every dim (``arr[::s1, ::s2, ...]``) — which mbo's
    lazy readers either reject or service pathologically (whole-file reads).
    This override reads ~64 whole 2D frames via fully scalar keys instead
    (see :func:`_sample_array`).

    It also repairs the window indexer's stop bound (see
    ``_get_slider_dims_indexer``).
    """

    def _get_slider_dims_indexer(self, indices):
        """Fix the exclusive stop bound of every windowed dim.

        Upstream clamps a window's stop with ``min(shape[dim] - 1, stop)``.
        A slice stop is exclusive, so at the last position of a windowed dim
        the slice collapses to ``slice(n - 1, n - 1)`` — empty. A numpy array
        then renders an all-NaN frame ("Mean of empty slice"); a lazy reader
        returns something ``np.asarray`` folds to a 0-d object array, and the
        fetch dies with "windowed_slice.ndim != len(spatial_dims): 0 != 2".

        Recompute the stop against the correct bound. Calling super() first
        keeps this a no-op once upstream clamps with ``shape[dim]``.
        """
        indexer = super()._get_slider_dims_indexer(indices)
        for dim in set(self.slider_dims) - set(self.spatial_dims):
            func, size = self.window_funcs.get(dim, (None, None))
            if func is None or size is None or dim not in self.window_order:
                continue
            stop = self.slider_dim_transforms[dim](indices[dim] + size / 2)
            start = indexer[dim].start
            indexer[dim] = slice(
                start, min(self.shape[dim], max(stop, start + 1)), 1
            )
        return indexer

    def _recompute_histogram(self):
        if not self._compute_histogram or self.data is None:
            self._histogram = None
            return

        sub = np.asarray(_sample_array(self.data)).ravel()
        if sub.size == 0:
            self._histogram = None
            return
        if sub.dtype.kind == "f":
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                self._histogram = None
                return
        elif sub.dtype.kind not in "biu":
            # non-numeric sample (object dtype from an exotic reader):
            # degrade to no histogram instead of raising mid-construction
            self._histogram = None
            return
        try:
            self._histogram = np.histogram(sub, bins=100)
        except (TypeError, ValueError):
            self._histogram = None


class _NDDataList(list):
    """list of the ORIGINAL data arrays; ``data[i] = arr`` performs the full
    mbo swap (re-derive dims, rebuild graphic/colorbar, clear funcs, reset
    indices)."""

    def __init__(self, viewer, items):
        super().__init__(items)
        self._viewer = viewer

    def __setitem__(self, i, new_array):
        self._viewer._replace_data(i, new_array)


class _MboIndicesView:
    """mbo indices contract over the ReferenceIndex.

    Name-keyed get/set, iteration yields index VALUES in slider order
    (consumers snapshot positions with ``list(iw.indices)``), ``len`` is the
    slider count. Values are 0-based ints; the reference space (sliders) is
    1-based."""

    def __init__(self, viewer: "MboNDViewer"):
        self._viewer = viewer

    def __getitem__(self, name):
        dim = self._viewer._resolve_dim(name)
        return int(round(float(self._viewer._ndw.indices[dim]))) - 1

    def __setitem__(self, name, value):
        dim = self._viewer._resolve_dim(name)
        value = self._viewer._check_index(dim, value)
        self._viewer._ndw.indices.set_dim_index(dim, value + 1)
        self._viewer._flush_index_fetches()

    def __iter__(self):
        ri = self._viewer._ndw.indices
        return iter(
            [int(round(float(ri[d]))) - 1 for d in self._viewer._dim_names]
        )

    def __len__(self):
        return len(self._viewer._dim_names)

    def __repr__(self):
        ri = self._viewer._ndw.indices
        return repr(
            {d: int(round(float(ri[d]))) - 1 for d in self._viewer._dim_names}
        )


class _DimStateView:
    """View over one per-dim NDWidgetUI state dict that accepts int
    positions (``_keyboard.toggle_playback`` indexes positionally), dim
    names, mbo display labels, and positional letters."""

    def __init__(self, viewer: "MboNDViewer", store: dict):
        self._viewer = viewer
        self._store = store

    def _key(self, key) -> str:
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            dims = self._viewer._dim_names
            if not 0 <= int(key) < len(dims):
                raise IndexError(key)
            return dims[int(key)]
        return self._viewer._resolve_dim(key)

    def __getitem__(self, key):
        return self._store[self._key(key)]

    def __setitem__(self, key, value):
        self._store[self._key(key)] = value

    def __contains__(self, key):
        try:
            return self._key(key) in self._store
        except (KeyError, IndexError):
            return False

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def __iter__(self):
        return iter(self._viewer._dim_names)

    def __len__(self):
        return len(self._viewer._dim_names)

    def __repr__(self):
        return repr({d: self._store.get(d) for d in self._viewer._dim_names})


class _MboSlidersUI:
    """Adapter over ``NDWidgetUI`` exposing the vendored
    ``ImageWidgetSliders`` surface the GUI touches.

    ``NDWidgetUI`` hardcodes fps 20 and has no metadata seeding or
    user-override tracking, so ``seed_fps`` writes its per-dim
    ``_fps``/``_frame_time`` state directly and tracks what it seeded to
    detect (and never override) an fps the user typed into the bar.
    """

    def __init__(self, viewer: "MboNDViewer"):
        self._viewer = viewer
        # dims whose fps the user typed explicitly — seeding never overrides
        self._user_fps: set[str] = set()
        # last value seed_fps wrote per dim, to tell seeds from user edits
        self._seeded_fps: dict[str, int] = {}
        # vendored contract is a single loop bool; NDWidgetUI keeps one per
        # dim, so the scalar fans out to every dim (and is re-applied to
        # dims (re)created by a data swap)
        self._loop_all = False

    @property
    def _ndui(self):
        return self._viewer._ndw._sliders_ui

    # --- per-dim playback state (int/name keyed views) -------------------

    @property
    def _playing(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._playing)

    @property
    def _fps(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._fps)

    @property
    def _frame_time(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._frame_time)

    @property
    def _last_frame_time(self) -> _DimStateView:
        return _DimStateView(self._viewer, self._ndui._last_frame_time)

    # --- loop -------------------------------------------------------------

    @property
    def _loop(self) -> bool:
        return self._loop_all

    @_loop.setter
    def _loop(self, value: bool):
        self._loop_all = bool(value)
        for d in self._viewer._dim_names:
            if d in self._ndui._loop:
                self._ndui._loop[d] = self._loop_all

    # --- fps seeding ------------------------------------------------------

    def seed_fps(self, dim: str, fps) -> None:
        """Seed the playback rate for a dim from data metadata.

        Ignores None/non-numeric/non-finite/<=0 values, clamps to the
        vendored slider's [1, 50] range, and never overrides an fps the
        user typed themselves.
        """
        if fps is None:
            return
        try:
            d = self._viewer._resolve_dim(dim)
        except KeyError:
            return
        if d in self._user_fps:
            return
        current = self._ndui._fps.get(d)
        if current is not None and current != 20 and current != self._seeded_fps.get(d):
            # NDWidgetUI's fps input doesn't record user edits; any value
            # that is neither the default nor our own seed must be one
            self._user_fps.add(d)
            return
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            return
        # nan slips past `<= 0` and int(round(nan)) raises; inf overflows
        if fps <= 0 or not math.isfinite(fps):
            return
        value = min(max(int(round(fps)), 1), 50)
        self._ndui._fps[d] = value
        self._ndui._frame_time[d] = 1 / value
        self._seeded_fps[d] = value

    # --- dim churn hooks (called by the viewer around data swaps) ---------

    def _snapshot_fps(self) -> tuple[dict, dict]:
        return dict(self._ndui._fps), dict(self._ndui._frame_time)

    def _after_dims_changed(self, fps_snapshot: tuple[dict, dict] | None):
        # restore fps for dims that survived the swap (push_dim reset them
        # to 20), then fan the loop flag out to the new dim set
        if fps_snapshot is not None:
            fps, ftime = fps_snapshot
            for d in self._viewer._dim_names:
                if d in fps and d in self._ndui._fps:
                    self._ndui._fps[d] = fps[d]
                    self._ndui._frame_time[d] = ftime[d]
        for d in self._viewer._dim_names:
            if d in self._ndui._loop:
                self._ndui._loop[d] = self._loop_all


class _MboNDWidgetUI(NDWidgetUI):
    """NDWidgetUI drawing integer sliders for integer ranges."""

    def update(self):
        now = perf_counter()

        for dim, current_index in self._ndwidget.indices:
            imgui.push_id(f"{self._id_counter}_{dim}")
            rr = self._ndwidget.ranges[dim]

            if self._playing[dim]:
                if imgui.button(label=fa.ICON_FA_PAUSE):
                    self._playing[dim] = False
                if now - self._last_frame_time[dim] >= self._frame_time[dim]:
                    self._set_index(dim, current_index + rr.step)
                    self._last_frame_time[dim] = now
            else:
                if imgui.button(label=fa.ICON_FA_PLAY):
                    self._last_frame_time[dim] = 0
                    self._playing[dim] = True

            imgui.same_line()
            if imgui.button(label=fa.ICON_FA_BACKWARD_STEP) and not self._playing[dim]:
                self._set_index(dim, current_index - rr.step)

            imgui.same_line()
            if imgui.button(label=fa.ICON_FA_FORWARD_STEP) and not self._playing[dim]:
                self._set_index(dim, current_index + rr.step)

            imgui.same_line()
            if imgui.button(label=fa.ICON_FA_STOP):
                self._playing[dim] = False
                self._last_frame_time[dim] = 0
                self._ndwidget.indices.set_dim_index(dim, rr.start)

            imgui.same_line()
            _, self._loop[dim] = imgui.checkbox(
                label=fa.ICON_FA_ROTATE, v=self._loop[dim]
            )
            if imgui.is_item_hovered(0):
                imgui.set_tooltip("loop playback")

            imgui.same_line()
            imgui.text("framerate :")
            imgui.same_line()
            imgui.set_next_item_width(100)
            fps_changed, value = imgui.input_int(
                label="fps", v=self._fps[dim], step_fast=5
            )
            if imgui.is_item_hovered(0):
                imgui.set_tooltip(
                    "framerate is approximate and less reliable as it approaches your monitor refresh rate"
                )
            if fps_changed:
                value = min(max(value, 1), 100)
                self._fps[dim] = value
                self._frame_time[dim] = 1 / value

            imgui.text(str(dim))
            imgui.same_line()
            imgui.set_next_item_width(self.width * 0.85)

            if isinstance(rr, RangeContinuous):
                if all(
                    float(v).is_integer() for v in (rr.start, rr.stop, rr.step)
                ):
                    changed, new_index = imgui.slider_int(
                        v=int(round(current_index)),
                        v_min=int(rr.start),
                        v_max=int(rr.stop - rr.step),
                        label=f"##{dim}",
                    )
                else:
                    changed, new_index = imgui.slider_float(
                        v=current_index,
                        v_min=rr.start,
                        v_max=rr.stop - rr.step,
                        label=f"##{dim}",
                    )

                if changed:
                    if now - self._last_slider_movement[dim] > rr.throttle:
                        self._ndwidget.indices.set_dim_index(
                            dim, new_index, cancel_awaiting=True
                        )
                        self._last_slider_movement[dim] = now
                elif imgui.is_item_hovered():
                    if imgui.is_key_pressed(imgui.Key.right_arrow):
                        self._set_index(dim, current_index + rr.step)
                    elif imgui.is_key_pressed(imgui.Key.left_arrow):
                        self._set_index(dim, current_index - rr.step)

            imgui.pop_id()

        if not self._collapsed:
            height = round(
                imgui.get_cursor_screen_pos().y
                - self.y
                + imgui.get_style().window_padding.y
            )
            if height != self.size:
                self.size = height


class MboNDViewer:
    """NDWidget wrapped in the mbo ImageWidget contract (see module docs)."""

    def __init__(
        self,
        data,
        names: Sequence[str] | None = None,
        slider_dim_names: Sequence[str] | None = None,
        window_funcs=None,
        window_sizes=None,
        frame_apply=None,
        figure_shape: tuple[int, int] | None = None,
        figure_kwargs: dict | None = None,
        histogram_widget: bool = True,
        rgb: bool | Sequence[bool] | None = None,
        cmap: str = "plasma",
        graphic_kwargs: dict | None = None,
    ):
        if isinstance(data, (list, tuple)):
            arrays = list(data)
        else:
            arrays = [data]
        if not arrays:
            raise ValueError("`data` must contain at least one array")
        for arr in arrays:
            for attr in ("shape", "ndim", "dtype", "__getitem__"):
                if not hasattr(arr, attr):
                    raise TypeError(
                        f"data arrays must be array-like with `{attr}`, got: {type(arr)}"
                    )

        # per-array rgb flags
        if rgb is None:
            self._rgb = [bool(getattr(a, "rgb", False)) for a in arrays]
        elif isinstance(rgb, bool):
            self._rgb = [rgb] * len(arrays)
        else:
            self._rgb = [bool(r) for r in rgb]
            if len(self._rgb) != len(arrays):
                raise ValueError("`rgb` must match the number of data arrays")

        self._arrays = arrays
        self._histogram_widget = bool(histogram_widget)
        self._closed = False

        # display labels are a plain writable attribute; mesc_units
        # re-stamps it after a unit swap
        self._slider_dim_names = (
            tuple(slider_dim_names) if slider_dim_names else None
        )

        # shared positional slider-dim names (the ReferenceIndex dims)
        counts = [self._n_slider_dims(a, r) for a, r in zip(arrays, self._rgb)]
        n_dims = max(counts) if counts else 0
        if self._slider_dim_names and len(self._slider_dim_names) == n_dims:
            requested = self._slider_dim_names
        else:
            requested = None
        if requested is None:
            # bare path: prefer the array's own labels when they cover every
            # slider axis; else _make_dim_names falls back to the canonical
            # per-rank letters (5D data is TCZYX -> 't','c','z')
            requested = self._labels_from_arrays(arrays, counts, n_dims)
        self._dim_names: list[str] = list(
            self._make_dim_names(n_dims, requested)
        )

        # 1-based reference space (sliders show 1..n); _ref_to_index maps to
        # 0-based array indices
        ref_ranges = {
            name: RangeContinuous(1, self._dim_size(j) + 1, 1)
            for j, name in enumerate(self._dim_names)
        }

        fig_kwargs = {"controller_ids": "sync", "names": list(names) if names else None}
        fig_kwargs.update(figure_kwargs or {})
        fig_kwargs["shape"] = (
            tuple(figure_shape)
            if figure_shape is not None
            else calculate_figure_shape(len(arrays))
        )

        self._ndw = NDWidget(ref_ranges=ref_ranges, **fig_kwargs)

        # int-slider UI; add_imgui_window replaces the existing bottom window
        ui = _MboNDWidgetUI(self._ndw)
        self._ndw.figure.add_imgui_window(
            ui,
            location="bottom",
            size=57 + 50 * len(self._ndw.indices),
            title="NDWidget controls",
        )
        self._ndw._sliders_ui = ui

        # offscreen canvases never register with the rendercanvas loop, so
        # scheduled fetch tasks can be cancelled by the loop's no-canvases
        # self-stop before they run — index changes then never render (and
        # the serial-fetch bookkeeping wedges). Detect it once; the index
        # setters render synchronously in that case.
        self._offscreen = "offscreen" in type(self._ndw.figure.canvas).__module__

        # graphic styling
        gk = dict(graphic_kwargs or {})
        vmin = gk.pop("vmin", None)
        vmax = gk.pop("vmax", None)
        gk.setdefault("cmap", cmap)

        # legacy func-state mirrors (for hasattr probes and getters)
        self._window_funcs_state = None
        self._window_sizes_state = tuple(window_sizes) if window_sizes else None
        self._frame_apply_state: dict[int, Callable] = {}
        self._spatial_func_state = None
        # the USER spatial func routed to each graphic (unwrapped); the
        # processor may hold a float32-cast wrapper around it instead
        self._routed_spatial: dict[int, Callable | None] = {}

        # one NDImage per array, in subplot order
        self._ndgraphics: list[NDImage] = []
        subplots = [self._ndw[s] for s in self._ndw.figure]
        for i, (arr, nd_subplot) in enumerate(zip(arrays, subplots)):
            gname = None
            if names is not None and i < len(names):
                gname = str(names[i])
            ndg = self._add_image(nd_subplot, arr, self._rgb[i], name=gname)
            self._ndgraphics.append(ndg)
            self._style_graphic(ndg, vmin=vmin, vmax=vmax, **gk)

        # window funcs: legacy dict {"t": (func, size)} or positional
        # (func, ...) matched with positional window_sizes
        if window_funcs is not None:
            self.window_funcs = window_funcs
        if frame_apply is not None:
            self.frame_apply = frame_apply

        self._sliders = _MboSlidersUI(self)
        # bare factory calls never went through preview_data's fps seeding —
        # seed playback rate from the data's own frame rate here
        self._seed_fps_from_data()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _n_slider_dims(arr, rgb: bool) -> int:
        return max(int(arr.ndim) - 2 - (1 if rgb else 0), 0)

    def _dim_size(self, j: int) -> int:
        """max size of positional slider axis ``j`` across arrays that have it"""
        size = 1
        for arr, rgb in zip(self._arrays, self._rgb):
            if self._n_slider_dims(arr, rgb) > j:
                size = max(size, int(arr.shape[j]))
        return size

    @staticmethod
    def _labels_from_arrays(arrays, counts, n_dims: int):
        """the arrays' own ``slider_dim_labels`` for the bare path, but only
        when they name every slider axis of a widest array — a MescArray
        with singleton axes reports labels for the non-singleton axes only,
        which cannot be mapped positionally, so None (-> canonical letters)"""
        if n_dims == 0:
            return None
        for arr, cnt in zip(arrays, counts):
            if cnt != n_dims:
                continue
            try:
                labels = getattr(arr, "slider_dim_labels", None)
            except Exception:
                continue
            if labels:
                labels = tuple(str(x) for x in labels)
                if len(labels) == n_dims:
                    return labels
        return None

    def _seed_fps_from_data(self):
        """Seed the t playback rate from the data's own frame rate.

        Mirrors ``preview_data._seed_playback_fps`` for the bare factory
        path (and data swaps), which never goes through the GUI widget.
        ``seed_fps`` clamps to the bar's [1, 50] and never overrides an fps
        the user typed. Best-effort: metadata probes must never break
        construction.
        """
        try:
            arr = self._arrays[0]
            fs = getattr(arr, "fs", None)
            if fs is None:
                from mbo_utilities.metadata import get_param

                fs = get_param(getattr(arr, "metadata", None), "fs")
            self._sliders.seed_fps("t", fs)
        except Exception:
            pass

    def _make_dim_names(
        self, n: int, requested: Sequence[str] | None
    ) -> tuple[str, ...]:
        """positional slider-dim names: the requested display names when they
        match the slider count, else the canonical per-rank letters (5D data
        is TCZYX -> 't','c','z'; 4D -> 't','z') — deduped and kept clear of
        the reserved spatial names"""
        if requested is not None and len(requested) == n:
            base = [str(x) for x in requested]
        else:
            base = list(_default_dim_letters(n))
        out, seen = [], set()
        for name in base:
            if name in (_ROW, _COL, _RGB):
                name = name + "_"
            while name in seen:
                name = name + "_"
            seen.add(name)
            out.append(name)
        return tuple(out)

    def _add_image(self, nd_subplot, arr, rgb: bool, name: str | None = None):
        k = self._n_slider_dims(arr, rgb)
        spatial = (_ROW, _COL) + ((_RGB,) if rgb else ())
        dims = tuple(self._dim_names[:k]) + spatial
        return nd_subplot.add_nd_image(
            data=arr,
            dims=dims,
            spatial_dims=spatial,
            rgb_dim=_RGB if rgb else None,
            compute_histogram=self._histogram_widget,
            processor_type=MboNDImageProcessor,
            # fresh dict — the setter mutates
            slider_dim_transforms={d: _ref_to_index for d in dims[:k]} or None,
            name=name or "mbo_image",
        )

    def _style_graphic(self, ndg, vmin=None, vmax=None, cmap=None, **extra):
        g = ndg.graphic
        if g is None:
            return
        if cmap is not None and getattr(g, "cmap", None) is not None:
            with contextlib.suppress(Exception):
                g.cmap = cmap
        for key, val in extra.items():
            with contextlib.suppress(Exception):
                setattr(g, key, val)
        if vmin is None and vmax is None:
            return
        cb = ndg.histogram_widget
        # order matters: widen through vmax first so a new vmin above the
        # old vmax is never momentarily inverted
        if cb is not None:
            if vmax is not None:
                cb.vmax = float(vmax)
            if vmin is not None:
                cb.vmin = float(vmin)
        else:
            if vmax is not None:
                g.vmax = float(vmax)
            if vmin is not None:
                g.vmin = float(vmin)

    # ------------------------------------------------------------------
    # name resolution
    # ------------------------------------------------------------------

    def _resolve_dim(self, name) -> str:
        """display name / letter / reference dim -> reference dim name"""
        name = str(name)
        dims = self._dim_names
        if name in dims:
            return name
        labels = tuple(self._slider_dim_names or ())
        if name in labels:
            pos = labels.index(name)
            if pos < len(dims):
                return dims[pos]
        if name in _SLIDER_LETTERS:
            pos = _SLIDER_LETTERS.index(name)
            if pos < len(dims):
                return dims[pos]
        raise KeyError(name)

    def _check_index(self, dim: str, value) -> int:
        """validate an index for a (resolved) reference dim.

        The upstream ReferenceIndex silently clamps; the vendored widget
        raised — negative indexing was "not supported" and out-of-range
        raised an IndexError naming the dim, so keep that contract.
        """
        value = int(value)
        if value < 0:
            raise IndexError(
                f"negative indexing is not supported (got {value} for dim "
                f"{dim!r})"
            )
        rr = self._ndw.indices.ref_ranges.get(dim)
        if isinstance(rr, RangeContinuous):
            size = int(rr.stop - rr.start)
            if value >= size:
                raise IndexError(
                    f"index {value} out of bounds for dim {dim!r} with size "
                    f"{size}"
                )
        return value

    # ------------------------------------------------------------------
    # core mbo surface
    # ------------------------------------------------------------------

    @property
    def ndwidget(self) -> NDWidget:
        """the wrapped NDWidget (escape hatch, not part of the mbo contract)"""
        return self._ndw

    @property
    def ndgraphics(self) -> tuple:
        return tuple(self._ndgraphics)

    @property
    def figure(self):
        """the real ImguiFigure — canvas/renderer/imgui_renderer/subplot
        access all pass through untouched"""
        return self._ndw.figure

    @property
    def data(self) -> _NDDataList:
        return _NDDataList(self, self._arrays)

    @property
    def graphics(self) -> list:
        """the ImageGraphics, one per managed array"""
        return [ndg.graphic for ndg in self._ndgraphics]

    @property
    def n_sliders(self) -> int:
        return len(self._dim_names)

    @property
    def slider_dims(self) -> list[str]:
        """positional letters, vendored-compatible (``'t' in iw.slider_dims``
        gates the legacy window-funcs path in preview_data)"""
        return [
            _SLIDER_LETTERS[j] if j < len(_SLIDER_LETTERS) else f"dim{j}"
            for j in range(len(self._dim_names))
        ]

    @property
    def indices(self) -> _MboIndicesView:
        return _MboIndicesView(self)

    @indices.setter
    def indices(self, value):
        if isinstance(value, dict):
            items = {self._resolve_dim(k): v for k, v in value.items()}
        else:
            items = dict(zip(self._dim_names, value))
        items = {d: self._check_index(d, v) for d, v in items.items()}
        # cancel_awaiting=False -> queued serial path, every frame renders;
        # `iw.indices = list(iw.indices)` is the GUI's force-refresh idiom
        self._ndw.indices.set({d: v + 1 for d, v in items.items()})
        self._flush_index_fetches()

    @property
    def current_index(self) -> dict[str, int]:
        ri = self._ndw.indices
        return {d: int(round(float(ri[d]))) - 1 for d in self._dim_names}

    @current_index.setter
    def current_index(self, value: dict):
        self.indices = dict(value)

    @property
    def _sliders_ui(self) -> _MboSlidersUI:
        return self._sliders

    @property
    def cmap(self) -> list[str]:
        return [getattr(g, "cmap", None) for g in self.graphics]

    @cmap.setter
    def cmap(self, value):
        if isinstance(value, str):
            values = [value] * len(self._ndgraphics)
        else:
            values = list(value)
        for ndg, name in zip(self._ndgraphics, values):
            g = ndg.graphic
            if g is None or getattr(g, "cmap", None) is None:
                # rgb(a) images have no cmap
                continue
            with contextlib.suppress(Exception):
                g.cmap = name

    # ------------------------------------------------------------------
    # window / spatial function routing
    # ------------------------------------------------------------------

    def _fold_positional(self, funcs, sizes) -> dict:
        folded = {}
        funcs = tuple(funcs)
        sizes = tuple(sizes) if sizes else ()
        for j, dim in enumerate(self._dim_names):
            if j < len(funcs) and funcs[j] is not None:
                size = 1
                if j < len(sizes) and sizes[j]:
                    size = sizes[j]
                folded[dim] = (funcs[j], size)
        return folded

    @property
    def window_funcs(self):
        return self._window_funcs_state

    @window_funcs.setter
    def window_funcs(self, value):
        """Accepts the legacy dict protocol ``{"t": (func, size)}`` (keys may
        also be display names), positional ``(func, ...)`` tuples matched
        with ``window_sizes``, or None to clear.

        Routed to per-graphic ``NDProcessor.window_funcs`` AND
        ``window_order`` — funcs for dims absent from ``window_order`` are
        silently ignored by fastplotlib. Sizes are in reference units, which
        with our step-1 ranges equal frame counts.
        """
        self._window_funcs_state = value
        if value is None:
            translated = {}
        elif isinstance(value, dict):
            translated = {
                self._resolve_dim(k): tuple(v) for k, v in value.items()
            }
        elif isinstance(value, (tuple, list)):
            translated = self._fold_positional(value, self._window_sizes_state)
        else:
            raise TypeError(
                "window_funcs must be None, a {dim: (func, size)} dict, or a "
                f"positional tuple of funcs; got: {type(value)}"
            )

        for i, ndg in enumerate(self._ndgraphics):
            wf = {
                d: t
                for d, t in translated.items()
                if d in ndg.processor.slider_dims and t and t[0] is not None
            }
            # fresh dict per graphic: the fpl setter mutates its argument
            ndg.processor.window_funcs = dict(wf) if wf else None
            ndg.processor.window_order = tuple(wf.keys()) if wf else None
            # window activation can flip the float32-texture requirement
            self._apply_spatial(i)
        self._force_render()

    @property
    def window_sizes(self):
        return self._window_sizes_state

    @window_sizes.setter
    def window_sizes(self, value):
        self._window_sizes_state = tuple(value) if value else None
        # re-apply positional funcs with the new sizes
        if isinstance(self._window_funcs_state, (tuple, list)):
            self.window_funcs = self._window_funcs_state

    @property
    def frame_apply(self) -> dict:
        return self._frame_apply_state

    @frame_apply.setter
    def frame_apply(self, value):
        """Legacy per-graphic post-processing ``{data_ix: func}`` — routed to
        the per-graphic NDProcessor ``spatial_func`` (applied after window
        funcs, before rendering). The processor-level setter is used so no
        histogram recompute is triggered (the vendored frame_apply didn't
        touch the histogram either — mean-sub z-scrubbing sets this often).
        """
        if value is None:
            mapping = {}
        elif callable(value):
            mapping = {i: value for i in range(len(self._ndgraphics))}
        elif isinstance(value, dict):
            mapping = dict(value)
        else:
            raise TypeError(
                "frame_apply must be a callable or a {data_index: callable} dict"
            )
        self._frame_apply_state = mapping
        for i in range(len(self._ndgraphics)):
            self._route_spatial(i, mapping.get(i))
        self._force_render()

    @property
    def spatial_func(self):
        return self._spatial_func_state

    @spatial_func.setter
    def spatial_func(self, value):
        """Extended-protocol spatial funcs: a single callable/None or one per
        graphic. Same routing as ``frame_apply`` (no histogram recompute)."""
        self._spatial_func_state = value
        if isinstance(value, (list, tuple)):
            funcs = list(value)
        else:
            funcs = [value] * len(self._ndgraphics)
        for i, func in enumerate(funcs[: len(self._ndgraphics)]):
            self._route_spatial(i, func)
        self._force_render()

    def _route_spatial(self, i: int, func):
        """record graphic i's USER spatial func and push it to the processor"""
        self._routed_spatial[i] = func
        self._apply_spatial(i)

    def _apply_spatial(self, i: int):
        """Push graphic i's routed spatial func to its processor.

        When window/spatial funcs are active over INTEGER data the func is
        wrapped in a float32 cast and the graphic's texture is upgraded to
        float — otherwise the funcs' float output is truncated into the
        integer texture (t-mean(5) displayed truncated ints).
        """
        ndg = self._ndgraphics[i]
        func = self._routed_spatial.get(i)
        if self._needs_float(i):
            ndg.processor.spatial_func = _with_float32_cast(func)
            self._ensure_float_texture(ndg)
        else:
            ndg.processor.spatial_func = func

    def _needs_float(self, i: int) -> bool:
        """True when graphic i renders func output over integer data"""
        ndg = self._ndgraphics[i]
        data = ndg.data
        if data is None:
            return False
        try:
            kind = np.dtype(data.dtype).kind
        except Exception:
            return False
        if kind not in "bui":
            return False
        if self._routed_spatial.get(i) is not None:
            return True
        wf = ndg.processor.window_funcs or {}
        order = ndg.processor.window_order or ()
        return any(
            d in order and (wf.get(d) or (None, None))[0] is not None
            for d in wf
        )

    def _ensure_float_texture(self, ndg):
        """recreate ndg's graphic if its texture is integer (float funcs are
        active), preserving contrast; the texture then stays float until the
        graphic is next rebuilt (e.g. a data swap)"""
        g = ndg.graphic
        if g is None:
            return
        try:
            kind = np.asarray(g.data.value).dtype.kind
        except Exception:
            return
        if kind not in "bui":
            return
        vmin = getattr(g, "vmin", None)
        vmax = getattr(g, "vmax", None)
        # _create_graphic re-fetches through the (now float-casting)
        # processor, carries cmap over, and rebinds the colorbar — but it
        # also resets vmin/vmax, so restore the snapshot after
        with contextlib.suppress(Exception):
            run_sync(ndg._create_graphic())
        cb = ndg.histogram_widget
        target = cb if cb is not None else ndg.graphic
        if target is not None and vmin is not None and vmax is not None:
            with contextlib.suppress(Exception):
                target.vmax = float(vmax)
                target.vmin = float(vmin)

    def _force_render(self):
        """synchronously refetch + redraw every graphic at the current index"""
        for ndg in self._ndgraphics:
            if ndg.data is None or ndg.graphic is None:
                continue
            with contextlib.suppress(Exception):
                run_sync(ndg._set_indices_())

    def _flush_index_fetches(self):
        """Deliver index changes synchronously on offscreen canvases.

        Offscreen canvases never register with the rendercanvas loop, so its
        no-canvases self-stop cancels scheduled fetch tasks; a task cancelled
        before its first step skips ``_fetch_request``'s finally block and
        leaves ``ReferenceIndex._fetch_request_active[ndg]`` True forever —
        every later serial fetch just queues behind a task that no longer
        exists. There is no reliable loop to wait on offscreen, so render
        synchronously and drop the queued request that was just serviced
        (in place — a still-scheduled task must find its queue, not a
        KeyError). Onscreen loops always have a registered canvas and keep
        the async path.
        """
        if not getattr(self, "_offscreen", False):
            return
        ri = self._ndw.indices
        for ndg in self._ndgraphics:
            queue = ri._fetch_request_queue.get(ndg)
            if queue is not None:
                queue.clear()
        self._force_render()

    # ------------------------------------------------------------------
    # contrast / histogram
    # ------------------------------------------------------------------

    @property
    def compute_histogram(self) -> bool:
        return self._histogram_widget

    @compute_histogram.setter
    def compute_histogram(self, value: bool):
        self._histogram_widget = bool(value)
        for ndg in self._ndgraphics:
            ndg.compute_histogram = self._histogram_widget

    def _set_contrast(self, ndg, block):
        """Rescale one graphic to the value range of ``block`` and redraw its
        colorbar. ``block`` is whatever sample the caller decided represents
        the data: the current frame, or a subsample of the whole array."""
        block = np.asarray(block)
        finite = block[np.isfinite(block)] if block.dtype.kind == "f" else block
        if finite.size == 0:
            return
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1.0

        cb = ndg.histogram_widget
        if cb is not None:
            # the histogram setter also re-derives the value axis the bar
            # spans, so the handles land inside it
            cb.histogram = np.histogram(finite.ravel(), bins=100)
            # order matters: vmax first so vmin > old vmax never inverts
            cb.vmax = vmax
            cb.vmin = vmin
        else:
            g = ndg.graphic
            if g is not None:
                g.vmin, g.vmax = vmin, vmax

    def reset_vmin_vmax(self):
        """Reset contrast w.r.t. the full data (bounded ~64-frame scalar-key
        sample across every scrollable axis, lazy-reader friendly)."""
        for ndg, arr in zip(self._ndgraphics, self._arrays):
            self._set_contrast(ndg, _sample_array(arr))

    def reset_vmin_vmax_frame(self):
        """Reset contrast + histogram w.r.t. the currently displayed frame
        (post window/spatial funcs)."""
        for ndg in self._ndgraphics:
            g = ndg.graphic
            if g is None:
                continue
            self._set_contrast(ndg, g.data.value)

    # ------------------------------------------------------------------
    # data swap
    # ------------------------------------------------------------------

    def _teardown_ndgraphic(self, ndg):
        """fully retire one NDGraphic: processor executor, graphic, colorbar,
        subplot registration, ReferenceIndex bookkeeping"""
        # already-scheduled fetches must not touch the dead graphic
        ndg._set_indices_ = _noop_indices
        ndg.pause = True
        with contextlib.suppress(Exception):
            ndg.processor.close()
        subplot = ndg._nd_subplot.subplot
        if ndg.graphic is not None:
            with contextlib.suppress(Exception):
                subplot.delete_graphic(ndg.graphic)
            ndg._graphic = None
        if getattr(ndg, "_histogram_widget", None) is not None:
            with contextlib.suppress(Exception):
                subplot.remove_imgui_window("right")
            ndg._histogram_widget = None
        with contextlib.suppress(ValueError):
            ndg._nd_subplot._nd_graphics.remove(ndg)
        ri = self._ndw.indices
        ri._fetch_rev.pop(ndg, None)
        # A scheduled/in-flight serial fetch task closes over ndg and starts
        # with `queue = ri._fetch_request_queue[ndg]` — popping the key here
        # KeyErrors inside the rendercanvas task, whose finally then
        # re-inserts a dead active key forever (pinning ndg + its array).
        # Clear the queue IN PLACE so the task drains nothing and cleans up
        # after itself; drop the keys now only when no task can be running.
        queue = ri._fetch_request_queue.get(ndg)
        if queue is not None:
            queue.clear()
        if not ri._fetch_request_active.get(ndg, False):
            # no coroutine scheduled or in flight for this graphic
            ri._fetch_request_queue.pop(ndg, None)
            ri._fetch_request_active.pop(ndg, None)
        # else: deferred to _sweep_dead_fetch_state (next swap / close)

    def _sweep_dead_fetch_state(self):
        """drop ReferenceIndex fetch bookkeeping for torn-down graphics.

        Keys whose fetch task was still scheduled at teardown time could not
        be removed then (the task itself indexes the queue dict); once the
        task has run it removed its own queue key and reset its active flag,
        so anything left False for a dead graphic is safe to drop here.
        """
        ri = self._ndw.indices
        live = set(self._ndw.ndgraphics)
        for store in (ri._fetch_request_active, ri._fetch_request_queue):
            for ndg in [g for g in store if g not in live]:
                if ri._fetch_request_active.get(ndg, False):
                    # task still pending: keep its keys, an empty queue is
                    # all it needs to finish cleanly on its next loop step
                    queue = ri._fetch_request_queue.get(ndg)
                    if queue is not None:
                        queue.clear()
                    continue
                ri._fetch_request_active.pop(ndg, None)
                ri._fetch_request_queue.pop(ndg, None)

    def _pop_ref_dim(self, name: str):
        """remove one dim from the ReferenceIndex + every registered
        NDWidget's slider UI (ReferenceIndex.pop_dim is an empty stub
        upstream, so the removal is done directly)"""
        ri = self._ndw.indices
        ri._ref_ranges.pop(name, None)
        ri._indices.pop(name, None)
        for ndw in ri._ndwidgets:
            if name in ndw._sliders_ui._playing:
                ndw._sliders_ui.pop_dim(name)

    def _replace_data(self, i, new_array):
        """``data[i] = new_array``: the FULL mbo swap.

        Re-derives dimensionality (rank may change 2D..5D — NDProcessor.dims
        is read-only, so the NDImage is recreated and the ReferenceIndex
        dims are reshaped), rebuilds histogram/colorbar, clears
        window/spatial funcs, resets indices to 0, re-seeds the playback fps
        from the new data, and shuts down the replaced graphic's processor
        executor. If installing the new array fails (a broken reader raises
        during histogram/first-frame reads), the old array is reinstalled so
        the viewer keeps working, and the failure is re-raised.
        """
        i = int(i)
        if not 0 <= i < len(self._arrays):
            raise IndexError(i)
        for attr in ("shape", "ndim", "dtype", "__getitem__"):
            if not hasattr(new_array, attr):
                raise TypeError(
                    f"replacement data must be array-like with `{attr}`"
                )

        # graphics torn down by previous swaps may finally have run their
        # scheduled fetch task; their deferred bookkeeping is dead now
        self._sweep_dead_fetch_state()

        old_ndg = self._ndgraphics[i]
        nd_subplot = old_ndg._nd_subplot
        old_array = self._arrays[i]
        old_rgb = self._rgb[i]

        # re-derive rgb only when the new array states it (vendored rule)
        new_rgb = getattr(new_array, "rgb", None)
        if isinstance(new_rgb, bool):
            self._rgb[i] = new_rgb

        old_cmap = None
        if old_ndg.graphic is not None:
            old_cmap = getattr(old_ndg.graphic, "cmap", None)

        self._teardown_ndgraphic(old_ndg)
        try:
            self._install_array(i, new_array, nd_subplot, old_ndg.name, old_cmap)
        except Exception:
            # failure-safe: reinstall the old (previously working) array so
            # the swap never leaves the viewer half-torn-down; indices and
            # funcs are reset, but scrubbing keeps working
            self._rgb[i] = old_rgb
            with contextlib.suppress(Exception):
                self._install_array(
                    i, old_array, nd_subplot, old_ndg.name, old_cmap
                )
            raise

    def _install_array(self, i, new_array, nd_subplot, name, cmap):
        """swap machinery shared by the forward path and the rollback path:
        reshape the shared dim space for ``new_array`` at data slot ``i``,
        clear stale func state, and rebuild the graphic + colorbar"""
        self._arrays[i] = new_array

        # ---- reshape the shared index space ----
        counts = [
            self._n_slider_dims(a, r) for a, r in zip(self._arrays, self._rgb)
        ]
        need = max(counts) if counts else 0
        ri = self._ndw.indices
        ndui = self._ndw._sliders_ui
        fps_snapshot = self._sliders._snapshot_fps() if hasattr(self, "_sliders") else None

        if len(self._arrays) == 1:
            # single-array viewer: rebuild the dim space wholesale so the
            # new array's own labels (mesc slider_dim_labels) name the dims
            new_names = self._derive_swap_names(new_array, need)
            for dim in list(self._dim_names):
                self._pop_ref_dim(dim)
            self._dim_names = list(new_names)
            ri.push_dims(
                {
                    dim: RangeContinuous(1, self._dim_size(j) + 1, 1)
                    for j, dim in enumerate(self._dim_names)
                }
            )
        else:
            # multi-array viewer: dims are positional and shared with the
            # other graphics — never rename existing positions
            while len(self._dim_names) > need:
                self._pop_ref_dim(self._dim_names.pop())
            for j, dim in enumerate(self._dim_names):
                # refresh the range in place; UI state survives
                ri._ref_ranges[dim] = RangeContinuous(1, self._dim_size(j) + 1, 1)
                ri._indices[dim] = 1
            while len(self._dim_names) < need:
                j = len(self._dim_names)
                dim = self._make_dim_names(j + 1, None)[j]
                while dim in self._dim_names or dim in ri.dims:
                    dim = dim + "_"
                self._dim_names.append(dim)
                ri.push_dims({dim: RangeContinuous(1, self._dim_size(j) + 1, 1)})

        # the display labels must track the (possibly re-keyed) dim space;
        # still a plain writable attr — consumer restamps keep working
        self._slider_dim_names = tuple(self._dim_names) or None

        if hasattr(self, "_sliders"):
            self._sliders._after_dims_changed(fps_snapshot)
            self._seed_fps_from_data()

        # ---- stale closures over the old data are invalid ----
        self._window_funcs_state = None
        self._window_sizes_state = None
        self._frame_apply_state = {}
        self._spatial_func_state = None
        self._routed_spatial = {}
        for j, ndg in enumerate(self._ndgraphics):
            if j == i:
                continue
            ndg.processor.window_funcs = None
            ndg.processor.window_order = None
            ndg.processor.spatial_func = None

        # start the new graphic at index 0 (ref 1) in every dim
        for dim in self._dim_names:
            ri._indices[dim] = 1

        # ---- rebuild the graphic + colorbar for the new array ----
        new_ndg = self._add_image(nd_subplot, new_array, self._rgb[i], name=name)
        self._ndgraphics[i] = new_ndg
        if cmap is not None:
            self._style_graphic(new_ndg, cmap=cmap)

        # contrast for the new dataset (vendored parity: reset on swap;
        # NDImage already ran graphic.reset_vmin_vmax on the first frame,
        # this widens it to the bounded full-data sample)
        with contextlib.suppress(Exception):
            self._set_contrast(new_ndg, _sample_array(new_array))

        # resize the playback bar to the new slider count (it also
        # auto-sizes on the next drawn frame)
        with contextlib.suppress(Exception):
            ndui.size = 57 + 50 * len(self._dim_names)

        # schedule a render of every graphic at the reset indices and fire
        # the indices handlers
        if self._dim_names:
            ri.set({d: 0 for d in self._dim_names})
            self._flush_index_fetches()
        else:
            self._force_render()

    def _derive_swap_names(self, arr, n: int) -> tuple[str, ...]:
        """slider-dim names for a swapped-in array: its own
        ``slider_dim_labels`` when they match the new slider count, else the
        current names when the count is unchanged, else letters."""
        labels = getattr(arr, "slider_dim_labels", None)
        if labels:
            labels = tuple(str(x) for x in labels)
            if len(labels) == n:
                return self._make_dim_names(n, labels)
        if len(self._dim_names) == n:
            return self._make_dim_names(n, self._dim_names)
        return self._make_dim_names(n, None)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def show(self, **kwargs):
        return self._ndw.show(**kwargs)

    def close(self):
        """Close the viewer.

        Shuts down every NDGraphic's processor executor (fastplotlib never
        does) and guards the offscreen case where ``Figure._output`` was
        never set (``Figure.close()`` raises AttributeError there).
        """
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._sweep_dead_fetch_state()
        for ndg in self._ndgraphics:
            ndg._set_indices_ = _noop_indices
            with contextlib.suppress(Exception):
                ndg.processor.close()
        fig = self._ndw.figure
        if getattr(fig, "_output", None) is not None:
            with contextlib.suppress(Exception):
                fig.close()
        else:
            # offscreen canvas: Figure.close() would crash on _output=None
            with contextlib.suppress(Exception):
                fig.canvas.close()

    def __repr__(self):
        shapes = [tuple(getattr(a, "shape", ())) for a in self._arrays]
        return (
            f"{type(self).__name__}(n_data={len(self._arrays)}, "
            f"dims={tuple(self._dim_names)}, shapes={shapes})"
        )
