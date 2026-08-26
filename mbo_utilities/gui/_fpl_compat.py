"""ImageWidget construction compat across fastplotlib variants.

The GUI was written against mbo-fastplotlib's extended ImageWidget (named
sliders via ``slider_dim_names``/``.indices``, split ``window_funcs``/
``window_sizes``). ``create_image_widget`` is the single construction seam
every viewer goes through (run_gui plus the arrays' ``imshow`` methods).

By default it now returns an :class:`mbo_utilities.gui._ndviewer.MboNDViewer`
— an adapter over the ndwidget branch's ``fastplotlib.NDWidget`` that
preserves the exact mbo ImageWidget contract (``.data`` swaps, name-keyed
``.indices``, ``.graphics``/``.n_sliders``/``._sliders_ui``/
``._slider_dim_names``, window/spatial func routing, contrast resets), so
consumer files stay untouched.

Set the environment variable ``MBO_LEGACY_IMAGE_WIDGET=1`` to select the
previous behavior instead: the vendored legacy ImageWidget copy at
``mbo_utilities.gui._vendor._widget`` (or a real ``fpl.ImageWidget`` when the
installed fastplotlib still ships one), with the extended kwargs translated
and the ``_slider_dim_names``/``.indices`` adapters bolted on. This is cheap
insurance while the NDWidget path beds in.

This module also owns the ``ImguiColorbar._draw_histogram`` monkeypatch for
imgui-bundle 1.92.5 (see :func:`ensure_colorbar_patch`) so BOTH paths get it
— without it the first histogram draw wedges all rendering.
"""

import inspect
import os

import fastplotlib as fpl

# upstream slider dims, left -> right (ALLOWED_SLIDER_DIMS in fpl)
_SLIDER_DIM_ORDER = ("t", "z", "c")


def image_widget_cls():
    """ImageWidget class across variants.

    mbo-fastplotlib and release fastplotlib export it at top level; the
    ndwidget branch keeps the module but broke it (removed tools), so the
    vendored copy patched onto that branch's APIs is the last resort.
    """
    cls = getattr(fpl, "ImageWidget", None)
    if cls is not None:
        return cls
    try:
        from fastplotlib.widgets.image_widget import ImageWidget

        return ImageWidget
    except ImportError:
        from mbo_utilities.gui._vendor._widget import ImageWidget

        return ImageWidget


def supports_named_sliders() -> bool:
    return (
        "slider_dim_names"
        in inspect.signature(image_widget_cls().__init__).parameters
    )


class _NamedIndices:
    """Adapter over the fixed "t"/"z" slider dims with mbo-fastplotlib's
    indices contract: name-keyed get/set, positional list assignment, and
    iteration yields the index VALUES in slider order (consumers do
    ``list(iw.indices)`` to snapshot positions)."""

    def __init__(self, iw):
        self._iw = iw

    def _ordered_dims(self) -> list[str]:
        return [d for d in _SLIDER_DIM_ORDER if d in self._iw.current_index]

    def _dim(self, name: str) -> str:
        current = self._iw.current_index
        if name in current:
            return name
        names = tuple(getattr(self._iw, "_slider_dim_names", None) or ())
        try:
            pos = names.index(name)
        except ValueError:
            raise KeyError(name) from None
        dims = self._ordered_dims()
        if pos >= len(dims):
            raise KeyError(name)
        return dims[pos]

    def __getitem__(self, name):
        return self._iw.current_index[self._dim(name)]

    def __setitem__(self, name, value):
        self._iw.current_index = {self._dim(name): int(value)}

    def __iter__(self):
        current = self._iw.current_index
        return iter([current[d] for d in self._ordered_dims()])

    def __len__(self):
        return len(self._iw.current_index)

    def __repr__(self):
        current = self._iw.current_index
        return repr({d: current[d] for d in self._ordered_dims()})


def _fold_window_funcs(names, funcs, sizes):
    """mbo positional (func, ...)/(size, ...) -> upstream {"t": (func, size)}."""
    folded = {}
    if not funcs:
        return None
    for i, dim in enumerate(_SLIDER_DIM_ORDER):
        if i < len(funcs) and funcs[i] is not None:
            size = 1
            if sizes and i < len(sizes) and sizes[i]:
                size = int(sizes[i])
            folded[dim] = (funcs[i], size)
    return folded or None


def _fixed_draw_histogram(self, draw_list, x_left, x_right, bar_y, bar_h):
    # upstream ImguiColorbar._draw_histogram calls
    # add_polyline(points, col, thickness, flags); imgui-bundle 1.92.5 binds
    # add_polyline(points, col, flags, thickness) — same body, fixed call
    from imgui_bundle import imgui

    counts, edges = self._histogram
    cmin = counts.min()
    cmax = counts.max()
    span = cmax - cmin
    if span <= 0:
        return

    color = imgui.color_convert_float4_to_u32((0.7, 0.7, 0.7, 1.0))
    hist_w = x_right - x_left
    if hist_w <= 0:
        return
    norm = (counts - cmin) / span
    centers = 0.5 * (edges[:-1] + edges[1:])
    points = [
        imgui.ImVec2(x_right - frac * hist_w, self._value_to_y(c, bar_y, bar_h))
        for frac, c in zip(norm, centers)
    ]
    draw_list.add_polyline(points, color, 0, 1.5)


_fixed_draw_histogram._mbo_polyline_patch = True


def ensure_colorbar_patch():
    """Replace ``ImguiColorbar._draw_histogram`` with a call-order-fixed copy.

    Upstream (fastplotlib/ui/_colorbar.py) passes ``(points, color,
    thickness, flags)`` to ``add_polyline`` but imgui-bundle 1.92.5 (pinned;
    wgpu's imgui backend needs <1.92.900) binds ``(points, col, flags,
    thickness)`` — the resulting per-frame TypeError trips imgui's "Missing
    EndChild()" assert and wedges every subsequent figure draw. NDImage's
    colorbars hit the same method, so BOTH the NDWidget path and the vendored
    legacy path need this. Idempotent; applied at module import and again
    defensively by ``MboNDViewer.__init__``.
    """
    from fastplotlib.ui import ImguiColorbar

    if getattr(ImguiColorbar._draw_histogram, "_mbo_polyline_patch", False):
        return
    ImguiColorbar._draw_histogram = _fixed_draw_histogram


ensure_colorbar_patch()


def create_image_widget(**kwargs):
    """Viewer factory accepting the extended mbo kwargs everywhere.

    Default: an NDWidget-backed :class:`~mbo_utilities.gui._ndviewer.MboNDViewer`
    preserving the mbo ImageWidget contract. With ``MBO_LEGACY_IMAGE_WIDGET=1``
    in the environment: the legacy (vendored) ImageWidget path, unchanged.

    Accepted kwargs (translated as needed on either path): ``data`` (a single
    array or a list), ``names``, ``slider_dim_names``, ``window_funcs``
    (positional tuple or legacy ``{"t": (func, size)}`` dict),
    ``window_sizes``, ``frame_apply``, ``figure_shape``, ``figure_kwargs``
    (incl. ``canvas``/``canvas_kwargs``/``size``), ``histogram_widget``,
    ``rgb``, ``cmap``, ``graphic_kwargs`` (``vmin``/``vmax`` respected).
    """
    if os.environ.get("MBO_LEGACY_IMAGE_WIDGET") == "1":
        return _create_legacy_image_widget(**kwargs)

    from mbo_utilities.gui._ndviewer import MboNDViewer

    return MboNDViewer(**kwargs)


def _create_legacy_image_widget(**kwargs):
    """The pre-NDWidget factory: a real/vendored ImageWidget instance."""
    cls = image_widget_cls()
    if supports_named_sliders():
        return cls(**kwargs)

    names = kwargs.pop("slider_dim_names", None)
    funcs = kwargs.pop("window_funcs", None)
    sizes = kwargs.pop("window_sizes", None)
    if isinstance(funcs, (tuple, list)) or funcs is None:
        kwargs["window_funcs"] = _fold_window_funcs(names, funcs, sizes)
    else:
        kwargs["window_funcs"] = funcs

    iw = cls(**kwargs)
    iw._slider_dim_names = tuple(names) if names else None
    # the vendored class defines `indices` as a property already
    if not isinstance(getattr(type(iw), "indices", None), property):
        iw.indices = _NamedIndices(iw)
    return iw
