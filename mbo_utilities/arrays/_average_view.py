"""Read-time temporal frame averaging over any 5D TCZYX lazy array.

Wraps a source array and presents non-overlapping bins of ``factor`` frames
along T, each the mean of its bin: ``T' = T // factor``, output frame ``t`` is
``source[t * factor : (t + 1) * factor]`` averaged. Only T is touched - C, Z,
Y and X pass straight through, so a spatial sub-key still reads only the
pixels it asks for.

This is the same shape of thing as ``PhaseCorrectedView``: a step applied on
read, before anything downstream sees the data, so wrapping the array once in
the viewer makes the display, the ROI traces, extraction, demixing and the
writers all work on averaged frames without any of them knowing. Unlike phase
correction it changes ``T``, so the size accessors come off ``_shape5d`` (via
``LazyArray``) rather than the source, and ``metadata["fs"]`` is divided by
the factor - every downstream window, detrend and trace axis is in seconds.

A trailing partial bin is dropped, so every frame really is the mean of
``factor`` frames.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from mbo_utilities.arrays._registration import _TCZYX, _validated_tczyx_shape
from mbo_utilities.lazy_array import LazyArray

__all__ = ["FrameAveragedView", "average_frames"]

# bins kept from the last reads, so scrubbing one frame at a time does not
# re-read `factor` source frames every time
_CACHE_BINS = 8


class FrameAveragedView(LazyArray):
    """5D TCZYX lazy view whose frames are means of ``factor`` source frames.

    Parameters
    ----------
    source : LazyArray
        The array to wrap; never modified. ``view.source`` gives it back.
    factor : int
        Frames per output frame. 1 is a passthrough (callers should use the
        source instead; :func:`average_frames` does that for you).
    dtype : {"source", "float32"}
        ``"source"`` (default) rounds the mean back to the source dtype, so
        the view is a drop-in for the int16 writers and the suite2p / masknmf
        readers behind them. ``"float32"`` keeps the fractional means.
    """

    def __init__(self, source, factor: int, *, dtype: str = "source"):
        factor = int(factor)
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        t, c, z, y, x = _validated_tczyx_shape(source)
        if t // factor < 1:
            raise ValueError(
                f"factor {factor} is larger than the {t} timepoints available"
            )
        if dtype not in ("source", "float32"):
            raise ValueError(f"dtype must be 'source' or 'float32', got {dtype!r}")

        self._source = source
        self._factor = factor
        self._out_dtype = np.float32 if dtype == "float32" else np.dtype(source.dtype)
        self._T, self._C, self._Z, self._Y, self._X = t // factor, c, z, y, x
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    @property
    def source(self):
        """The wrapped source array (never modified)."""
        return self._source

    @property
    def _arr(self):
        """The wrapped source, for one-level `_arr` unwrapping by callers."""
        return self._source

    @property
    def factor(self) -> int:
        """Source frames averaged into each frame of this view."""
        return self._factor

    @property
    def dtype(self):
        return self._out_dtype

    @property
    def dims(self) -> tuple[str, ...]:
        return _TCZYX

    def _shape5d(self) -> tuple[int, int, int, int, int]:
        return (self._T, self._C, self._Z, self._Y, self._X)

    def __len__(self) -> int:
        return self._T

    @property
    def metadata(self) -> dict:
        """The source's metadata with every frame-rate / frame-interval alias
        retimed and the frame count scaled, plus a history entry, so
        downstream code reads real seconds."""
        from mbo_utilities.metadata import scale_frame_rate

        meta = scale_frame_rate(
            dict(getattr(self._source, "metadata", None) or {}), self._factor
        )
        if isinstance(meta.get("num_frames"), (int, float)):
            meta["num_frames"] = self._T
        meta["frame_average"] = self._factor
        meta["processing_history"] = [
            *(meta.get("processing_history") or []),
            {"step": "frame_average", "factor": self._factor},
        ]
        return meta

    @metadata.setter
    def metadata(self, value):
        self._source.metadata = value

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def _key5(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if Ellipsis in key:
            i = key.index(Ellipsis)
            n_missing = 5 - (len(key) - 1)
            key = key[:i] + (slice(None),) * max(n_missing, 0) + key[i + 1 :]
        if len(key) > 5:
            raise IndexError(f"too many indices for 5D array: {len(key)}")
        return key + (slice(None),) * (5 - len(key))

    def _means(self, block: np.ndarray, n_bins: int) -> np.ndarray:
        """``(n_bins * factor, ...)`` source frames -> ``(n_bins, ...)`` means."""
        mean = block.reshape(n_bins, self._factor, *block.shape[1:]).mean(
            axis=1, dtype=np.float32
        )
        if self._out_dtype == np.float32:
            return mean
        if np.issubdtype(self._out_dtype, np.integer):
            mean = np.rint(mean)
        return mean.astype(self._out_dtype)

    def _whole_frame(self, rest) -> bool:
        return all(k == slice(None) for k in rest)

    def _bin(self, t: int, rest) -> np.ndarray:
        """One averaged frame, from the cache when it is a whole one."""
        whole = self._whole_frame(rest)
        if whole:
            cached = self._cache.get(t)
            if cached is not None:
                self._cache.move_to_end(t)
                return cached
        lo = t * self._factor
        block = np.asarray(self._source[(slice(lo, lo + self._factor),) + rest])
        out = self._means(block, 1)[0]
        if whole:
            self._cache[t] = out
            self._cache.move_to_end(t)
            while len(self._cache) > _CACHE_BINS:
                self._cache.popitem(last=False)
        return out

    def __getitem__(self, key):
        t_key, *rest = self._key5(key)
        rest = tuple(rest)

        if isinstance(t_key, (int, np.integer)):
            t = int(t_key)
            if t < 0:
                t += self._T
            if not 0 <= t < self._T:
                raise IndexError(f"t index {t_key} out of range for {self._T} frames")
            return self._bin(t, rest)

        if isinstance(t_key, slice):
            start, stop, step = t_key.indices(self._T)
            bins = list(range(start, stop, step))
            if bins and step == 1:
                # contiguous: one source read, then bin-average in place
                lo, hi = start * self._factor, stop * self._factor
                block = np.asarray(self._source[(slice(lo, hi),) + rest])
                return self._means(block, len(bins))
        else:
            bins = [int(t) if int(t) >= 0 else self._T + int(t) for t in np.ravel(t_key)]

        if not bins:
            # keep the trailing axes so an empty read still stacks/reshapes
            probe = np.asarray(self._bin(0, rest))
            return np.empty((0, *probe.shape), self._out_dtype)
        return np.stack([self._bin(t, rest) for t in bins])

    def __array__(self, dtype=None, copy=None):
        data = np.asarray(self[:])
        return data.astype(dtype) if dtype is not None else data

    def astype(self, dtype, *args, **kwargs):
        return np.asarray(self).astype(dtype, *args, **kwargs)

    def __getattr__(self, name):
        # forward domain attributes (filenames, source_path, roi, ...) to the
        # source. Everything T-shaped is defined above or comes off _shape5d,
        # so the source's frame count never leaks through here; underscore
        # names are not forwarded so __init__ stays recursion-safe.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_source"), name)

    def _imwrite(self, outpath, **kwargs):
        """Stream this view to disk; the averaging is baked into the output."""
        from mbo_utilities.arrays._base import _imwrite_base

        return _imwrite_base(self, outpath, **kwargs)

    def save(self, outpath, **kwargs):
        return self._imwrite(outpath, **kwargs)

    def __repr__(self) -> str:
        return (
            f"FrameAveragedView(shape={self.shape}, dtype={self.dtype}, "
            f"factor={self._factor}, source={type(self._source).__name__})"
        )


def average_frames(source, factor: int, *, dtype: str = "source"):
    """``source`` with every ``factor`` timepoints averaged into one.

    Returns the source unchanged for ``factor <= 1``, and unwraps an existing
    view rather than stacking a second one, so toggling in the GUI never
    compounds.
    """
    if isinstance(source, FrameAveragedView):
        source = source.source
    if int(factor) <= 1:
        return source
    return FrameAveragedView(source, factor, dtype=dtype)
