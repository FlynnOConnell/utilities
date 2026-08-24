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
from collections.abc import MutableMapping

import fastplotlib as fpl

# upstream slider dims, left -> right (ALLOWED_SLIDER_DIMS in fpl)
_SLIDER_DIM_ORDER = ("t", "z")


def image_widget_cls():
    """ImageWidget class across variants — the ndwidget branch drops the
    top-level export but keeps fastplotlib.widgets.image_widget.ImageWidget."""
    cls = getattr(fpl, "ImageWidget", None)
    if cls is None:
        from fastplotlib.widgets.image_widget import ImageWidget as cls
    return cls


def supports_named_sliders() -> bool:
    return (
        "slider_dim_names"
        in inspect.signature(image_widget_cls().__init__).parameters
    )


class _NamedIndices(MutableMapping):
    """name -> current_index adapter over fixed "t"/"z" slider dims."""

    def __init__(self, iw):
        self._iw = iw

    def _dim(self, name: str) -> str:
        current = self._iw.current_index
        if name in current:
            return name
        names = tuple(getattr(self._iw, "_slider_dim_names", None) or ())
        try:
            pos = names.index(name)
        except ValueError:
            raise KeyError(name) from None
        dims = [d for d in _SLIDER_DIM_ORDER if d in current]
        if pos >= len(dims):
            raise KeyError(name)
        return dims[pos]

    def __getitem__(self, name):
        return self._iw.current_index[self._dim(name)]

    def __setitem__(self, name, value):
        self._iw.current_index = {self._dim(name): int(value)}

    def __delitem__(self, name):
        raise TypeError("slider dims cannot be removed")

    def __iter__(self):
        names = tuple(getattr(self._iw, "_slider_dim_names", None) or ())
        return iter(names or self._iw.current_index)

    def __len__(self):
        return len(self._iw.current_index)


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
    iw.indices = _NamedIndices(iw)
    return iw
