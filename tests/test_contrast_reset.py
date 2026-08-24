"""
Contrast-reset and slider-count contracts for the vendored ImageWidget.

Both reset paths (the playback bar's two buttons, the ``v`` shortcut, and
auto-contrast-on-z) used to look for a ``histogram_lut`` in the subplot's
right dock. This build renders the LUT with `ImguiColorbar` instead, so that
dock never exists and every one of those controls silently did nothing. These
tests pin the replacement: the reset must actually move vmin/vmax onto the
data, and the value sample it derives them from must stay bounded on a movie
too large to read whole.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.gui._vendor._widget import (
    ALLOWED_SLIDER_DIMS,
    SCROLLABLE_DIMS_ORDER,
    SLIDER_UI_SIZES,
    ImageWidget,
    _sample_array,
)


class CountingArray:
    """Array-like that records every key it is indexed with."""

    def __init__(self, shape, dtype=np.uint16):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = np.dtype(dtype)
        self._raw = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
        self.keys = []

    def __getitem__(self, key):
        self.keys.append(key)
        return self._raw[key]


# ============================================================
# sampling
# ============================================================

class TestSampleArray:
    def test_two_dimensional_data_is_used_whole(self):
        data = np.arange(24, dtype=np.uint16).reshape(4, 6)
        assert np.array_equal(_sample_array(data), data)

    def test_reads_only_flat_frames_never_whole_volumes(self):
        """A partial key would pull a full volume per sample."""
        data = CountingArray((500, 4, 8, 32, 32))
        _sample_array(data)
        assert data.keys, "expected reads"
        for key in data.keys:
            assert isinstance(key, tuple) and len(key) == 3, key
            assert all(isinstance(i, int) for i in key), key

    @pytest.mark.parametrize(
        "shape",
        [(45046, 34, 34), (45046, 2, 34, 34), (500, 4, 8, 16, 16)],
    )
    def test_read_count_is_bounded_regardless_of_movie_length(self, shape):
        data = CountingArray(shape)
        _sample_array(data)
        assert len(data.keys) <= 72, len(data.keys)

    def test_sample_spans_every_scrollable_axis(self):
        """Sampling time alone reports one channel's range as the whole array's."""
        data = np.zeros((40, 3, 8, 8), dtype=np.uint16)
        data[:, 2] = 5000  # only the last channel is bright
        sample = _sample_array(data)
        assert sample.max() == 5000

    def test_sample_spans_the_whole_recording(self):
        data = np.zeros((400, 8, 8), dtype=np.uint16)
        data[399] = 9000  # only the last frame is bright
        assert _sample_array(data).max() == 9000

    def test_empty_axis_yields_an_empty_sample(self):
        assert _sample_array(np.zeros((0, 8, 8))).size == 0


# ============================================================
# contrast application
# ============================================================

class FakeGraphic:
    def __init__(self):
        self.vmin, self.vmax = -100.0, 4000.0


class FakeColorbar:
    """Mirrors the ImguiColorbar contract used by the reset."""

    def __init__(self, graphic):
        self._graphic = graphic
        self.histogram = None
        self._vmin, self._vmax = graphic.vmin, graphic.vmax

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, value):
        self._vmin = float(value)
        self._graphic.vmin = float(value)

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, value):
        self._vmax = float(value)
        self._graphic.vmax = float(value)


class FakeWidget:
    """Just enough ImageWidget surface for `_set_contrast`."""

    def __init__(self, histogram_widget=True):
        self.graphic = FakeGraphic()
        self._histogram_widget = histogram_widget
        self._colorbars = [FakeColorbar(self.graphic)] if histogram_widget else []
        self.subplot = {"image_widget_managed": self.graphic}

    def apply(self, block):
        ImageWidget._set_contrast(self, 0, self.subplot, block)


class TestSetContrast:
    def test_limits_move_onto_the_data(self):
        w = FakeWidget()
        w.apply(np.array([[10, 20], [30, 900]], dtype=np.uint16))
        assert (w.graphic.vmin, w.graphic.vmax) == (10.0, 900.0)

    def test_colorbar_histogram_tracks_the_reset(self):
        w = FakeWidget()
        w.apply(np.array([[10, 900]], dtype=np.uint16))
        counts, edges = w._colorbars[0].histogram
        # the histogram setter re-derives the value axis the bar spans
        assert (float(edges[0]), float(edges[-1])) == (10.0, 900.0)
        assert len(edges) == len(counts) + 1
        assert (w._colorbars[0].vmin, w._colorbars[0].vmax) == (10.0, 900.0)

    def test_works_without_a_colorbar(self):
        w = FakeWidget(histogram_widget=False)
        w.apply(np.array([[5, 50]], dtype=np.uint16))
        assert (w.graphic.vmin, w.graphic.vmax) == (5.0, 50.0)

    def test_flat_data_keeps_a_non_degenerate_range(self):
        w = FakeWidget()
        w.apply(np.full((4, 4), 7, dtype=np.uint16))
        assert w.graphic.vmin == 7.0
        assert w.graphic.vmax > w.graphic.vmin

    def test_all_nan_data_is_left_alone(self):
        w = FakeWidget()
        before = (w.graphic.vmin, w.graphic.vmax)
        w.apply(np.full((4, 4), np.nan))
        assert (w.graphic.vmin, w.graphic.vmax) == before

    def test_non_finite_values_do_not_poison_the_range(self):
        w = FakeWidget()
        w.apply(np.array([[1.0, np.inf], [np.nan, 40.0]]))
        assert (w.graphic.vmin, w.graphic.vmax) == (1.0, 40.0)


# ============================================================
# third slider
# ============================================================

class TestThreeScrollableDims:
    def test_a_third_axis_is_scrollable(self):
        assert SCROLLABLE_DIMS_ORDER[3] == "tzc"
        assert ALLOWED_SLIDER_DIMS[2] == "c"

    @pytest.mark.parametrize("rgb", [False, True])
    def test_five_dimensional_arrays_are_accepted(self, rgb):
        shape = (6, 2, 4, 32, 24) + ((3,) if rgb else ())
        n = ImageWidget._get_n_scrollable_dims(None, np.zeros(shape), rgb)
        assert n == 3

    def test_six_scrollable_axes_are_still_refused(self):
        with pytest.raises(ValueError, match="not supported"):
            ImageWidget._get_n_scrollable_dims(None, np.zeros((2, 2, 2, 2, 8, 8)), False)

    def test_playback_bar_has_a_height_for_every_slider_count(self):
        for n in range(len(ALLOWED_SLIDER_DIMS) + 1):
            assert SLIDER_UI_SIZES[n] > 0
        assert SLIDER_UI_SIZES[3] > SLIDER_UI_SIZES[2]

    def test_slider_dim_order_covers_every_allowed_dim(self):
        from mbo_utilities.gui._fpl_compat import _SLIDER_DIM_ORDER

        assert set(_SLIDER_DIM_ORDER) == set(ALLOWED_SLIDER_DIMS.values())
        # order must match the axis positions, not just the membership
        assert _SLIDER_DIM_ORDER == tuple(
            ALLOWED_SLIDER_DIMS[i] for i in sorted(ALLOWED_SLIDER_DIMS)
        )


class TestSliderLabels:
    """The playback bar shows the array's own axis names, not t/z/c."""

    @staticmethod
    def _bar(names, dims):
        from mbo_utilities.gui._vendor._sliders import ImageWidgetSliders

        bar = ImageWidgetSliders.__new__(ImageWidgetSliders)
        bar._image_widget = type(
            "IW", (), {"_slider_dim_names": names, "slider_dims": dims}
        )()
        return bar

    def test_named_axes_are_shown(self):
        bar = self._bar(("Timepoint", "Channel", "ROI"), ["t", "z", "c"])
        assert [bar._dim_label(d) for d in ("t", "z", "c")] == [
            "Timepoint", "Channel", "ROI",
        ]

    def test_falls_back_to_the_internal_letter(self):
        assert self._bar(None, ["t", "z"])._dim_label("z") == "z"
        assert self._bar(("Timepoint",), ["t", "z"])._dim_label("z") == "z"

    def test_playback_state_accepts_any_dim(self):
        """Fixed {"t", "z"} dicts KeyError'd on a third dim or a data swap."""
        import inspect

        from mbo_utilities.gui._vendor import _sliders

        src = inspect.getsource(_sliders.ImageWidgetSliders.__init__)
        for attr in ("_playing", "_fps", "_frame_time", "_last_frame_time"):
            line = next(ln for ln in src.splitlines() if f"self.{attr}" in ln)
            assert "defaultdict" in line, line
