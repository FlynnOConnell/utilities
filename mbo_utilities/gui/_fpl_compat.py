"""ImageWidget construction compat across fastplotlib variants.

The GUI was written against mbo-fastplotlib's extended ImageWidget (named
sliders via ``slider_dim_names``/``.indices``, split ``window_funcs``/
``window_sizes``). The upstream ``ndwidget`` branch — required by masknmf —
predates those APIs: sliders are the fixed "t"/"z" dims and window functions
travel as ``{"t": (func, size)}``. ``create_image_widget`` passes the extended
kwargs through when the installed fastplotlib supports them and otherwise
translates them, then bolts ``_slider_dim_names`` and a name->dim ``.indices``
adapter onto the instance so GUI call sites work unchanged.
"""

import inspect

import fastplotlib as fpl

# upstream slider dims, left -> right (ALLOWED_SLIDER_DIMS in fpl)
_SLIDER_DIM_ORDER = ("t", "z")


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


def create_image_widget(**kwargs):
    """ImageWidget factory accepting the extended mbo kwargs everywhere."""
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
