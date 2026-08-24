"""Movie views over a saved ``demixing_results.hdf5``.

Wraps masknmf's lazy component arrays (signal, residual, colorful, ...) in a
numpy-like TYX / TYXC interface the viewer's ImageWidget can display directly.
"""

from __future__ import annotations

import numpy as np


class MasknmfView:
    """Numpy-like view over one masknmf component array.

    masknmf arrays keep the t axis for scalar indices ((1, Y, X) for ``a[5]``)
    and only index along t; this normalizes both so the array slots into the
    ImageWidget's frame-slicing contract. ``rgb`` marks TYXC color arrays.
    """

    def __init__(self, arr, name: str, rgb: bool = False):
        self._arr = arr
        self.name = name
        self.rgb = rgb

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self._arr.shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return np.dtype(getattr(self._arr, "dtype", np.float32))

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            tkey, rest = key[0], tuple(key[1:])
        else:
            tkey, rest = key, ()
        if isinstance(tkey, range):
            tkey = slice(tkey.start, tkey.stop, tkey.step)
        scalar = isinstance(tkey, (int, np.integer))
        if scalar:
            tkey = int(tkey)
        out = np.asarray(self._arr[tkey])
        if scalar and out.ndim == self.ndim:
            out = out[0]
        if rest:
            out = out[rest] if scalar else out[(slice(None),) + rest]
        return out

    def __array__(self, dtype=None):
        out = np.asarray(self._arr[:])
        return out.astype(dtype) if dtype is not None else out


# (label, DemixingResults attribute, rgb)
DEMIX_VIEWS = (
    ("Signal (AC)", "ac_array", False),
    ("Colorful", "colorful_ac_array", True),
    ("Residual", "residual_array", False),
    ("Background", "fluctuating_background_array", False),
    ("PMD", "pmd_array", False),
    ("Trend", "trend_array", False),
)


def load_demix_views(path) -> dict[str, MasknmfView]:
    """Load ``demixing_results.hdf5`` and return {label: view} for every
    component array that resolves."""
    import masknmf

    results = masknmf.DemixingResults.from_hdf5(str(path))
    views: dict[str, MasknmfView] = {}
    for label, attr, rgb in DEMIX_VIEWS:
        try:
            arr = getattr(results, attr)
            if arr is None or len(arr.shape) < 3:
                continue
            views[label] = MasknmfView(arr, label, rgb=rgb)
        except Exception:
            continue
    return views
