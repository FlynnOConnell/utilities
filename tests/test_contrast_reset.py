"""
Contrast-reset and slider-count contracts for the viewer.

Both reset paths (the playback bar's buttons, the ``v`` shortcut, and
auto-contrast-on-z) must actually move vmin/vmax onto the data, and the
value sample they derive them from must stay bounded on a movie too large
to read whole.

The viewer is the NDWidget-backed ``MboNDViewer``
(mbo_utilities/gui/_ndviewer.py); these tests pin its ``_sample_array`` /
``_set_contrast`` / slider-dim derivation. The full rendering-level
contract battery is in tests/test_ndviewer.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from mbo_utilities.gui._ndviewer import (
    MboNDViewer,
    VMINMAX_SAMPLE_COUNTS,
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

    def test_sample_counts_cover_one_to_three_scroll_axes(self):
        for n in (1, 2, 3):
            counts = VMINMAX_SAMPLE_COUNTS[n]
            assert len(counts) == n
            assert int(np.prod(counts)) <= 72


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


class FakeNDG:
    """Just enough NDImage surface for ``MboNDViewer._set_contrast``."""

    def __init__(self, histogram_widget=True):
        self.graphic = FakeGraphic()
        self.histogram_widget = (
            FakeColorbar(self.graphic) if histogram_widget else None
        )

    def apply(self, block):
        # _set_contrast only touches the ndg passed in, never self/viewer
        MboNDViewer._set_contrast(object.__new__(MboNDViewer), self, block)


class TestSetContrast:
    def test_limits_move_onto_the_data(self):
        ndg = FakeNDG()
        ndg.apply(np.array([[10, 20], [30, 900]], dtype=np.uint16))
        assert (ndg.graphic.vmin, ndg.graphic.vmax) == (10.0, 900.0)

    def test_colorbar_histogram_tracks_the_reset(self):
        ndg = FakeNDG()
        ndg.apply(np.array([[10, 900]], dtype=np.uint16))
        counts, edges = ndg.histogram_widget.histogram
        # the histogram setter re-derives the value axis the bar spans
        assert (float(edges[0]), float(edges[-1])) == (10.0, 900.0)
        assert len(edges) == len(counts) + 1
        assert (ndg.histogram_widget.vmin, ndg.histogram_widget.vmax) == (
            10.0,
            900.0,
        )

    def test_works_without_a_colorbar(self):
        ndg = FakeNDG(histogram_widget=False)
        ndg.apply(np.array([[5, 50]], dtype=np.uint16))
        assert (ndg.graphic.vmin, ndg.graphic.vmax) == (5.0, 50.0)

    def test_flat_data_keeps_a_non_degenerate_range(self):
        ndg = FakeNDG()
        ndg.apply(np.full((4, 4), 7, dtype=np.uint16))
        assert ndg.graphic.vmin == 7.0
        assert ndg.graphic.vmax > ndg.graphic.vmin

    def test_all_nan_data_is_left_alone(self):
        ndg = FakeNDG()
        before = (ndg.graphic.vmin, ndg.graphic.vmax)
        ndg.apply(np.full((4, 4), np.nan))
        assert (ndg.graphic.vmin, ndg.graphic.vmax) == before

    def test_non_finite_values_do_not_poison_the_range(self):
        ndg = FakeNDG()
        ndg.apply(np.array([[1.0, np.inf], [np.nan, 40.0]]))
        assert (ndg.graphic.vmin, ndg.graphic.vmax) == (1.0, 40.0)


# ============================================================
# slider-dim derivation (adapter)
# ============================================================

class _ShapeOnly:
    def __init__(self, ndim):
        self.ndim = ndim


class TestSliderDims:
    """Scrollable-axis counting on the adapter. NDWidget has no hard cap on
    slider dims, so the vendored "six axes refused" ValueError is gone —
    a 6D array simply gets a fourth slider with a generated name."""

    def test_a_third_axis_is_scrollable(self):
        assert MboNDViewer._n_slider_dims(_ShapeOnly(5), rgb=False) == 3

    @pytest.mark.parametrize("rgb", [False, True])
    def test_five_dimensional_arrays_are_accepted(self, rgb):
        ndim = 5 + (1 if rgb else 0)
        assert MboNDViewer._n_slider_dims(_ShapeOnly(ndim), rgb) == 3

    def test_two_dimensional_arrays_have_no_sliders(self):
        assert MboNDViewer._n_slider_dims(_ShapeOnly(2), rgb=False) == 0

    def test_six_scrollable_axes_get_a_generated_name(self):
        # concept change vs the vendored widget, which raised ValueError:
        # NDWidget supports arbitrary dims, so a 6D array is accepted.
        # Unnamed letters follow the canonical mbo axis order — 5D data is
        # (T, C, Z, Y, X), so axis 1 is 'c' and axis 2 is 'z' (the vendored
        # positional order t/z/c put the 'z' slider on the C axis)
        assert MboNDViewer._n_slider_dims(_ShapeOnly(6), rgb=False) == 4
        names = MboNDViewer._make_dim_names(None, 4, None)
        assert names == ("t", "c", "z", "dim3")


class TestSliderLabels:
    """The playback bar shows the array's own axis names. On the adapter
    the ReferenceIndex dims ARE the display names, so NDWidgetUI labels
    sliders directly; these pin the name derivation + resolution rules."""

    @staticmethod
    def _viewer(dim_names, labels=None):
        v = object.__new__(MboNDViewer)
        v._dim_names = list(dim_names)
        v._slider_dim_names = tuple(labels) if labels else None
        return v

    def test_display_names_used_when_count_matches(self):
        names = MboNDViewer._make_dim_names(
            None, 3, ("Timepoint", "Channel", "ROI")
        )
        assert names == ("Timepoint", "Channel", "ROI")

    def test_falls_back_to_letters_on_count_mismatch(self):
        assert MboNDViewer._make_dim_names(None, 2, ("Timepoint",)) is not None
        assert MboNDViewer._make_dim_names(None, 2, None) == ("t", "z")

    def test_duplicate_names_are_deduped(self):
        assert MboNDViewer._make_dim_names(None, 2, ("a", "a")) == ("a", "a_")

    def test_reserved_spatial_names_are_mangled(self):
        (name,) = MboNDViewer._make_dim_names(None, 1, ("__row__",))
        assert name != "__row__"

    def test_resolution_order_ref_dim_then_label_then_letter(self):
        v = self._viewer(
            ["Timepoint", "Channel", "Z-plane"],
            labels=("Timepoint", "Channel", "Z-plane"),
        )
        assert v._resolve_dim("Timepoint") == "Timepoint"  # ref dim
        assert v._resolve_dim("Channel") == "Channel"  # label, positional
        assert v._resolve_dim("t") == "Timepoint"  # letter, positional
        assert v._resolve_dim("z") == "Channel"
        assert v._resolve_dim("c") == "Z-plane"
        with pytest.raises(KeyError):
            v._resolve_dim("nope")

    def test_stale_labels_resolve_positionally(self):
        # after a swap the ref dims may be letters while _slider_dim_names
        # still carries the previous display labels — those must keep
        # resolving by position rather than KeyError
        v = self._viewer(["t", "z"], labels=("Timepoint", "Z-plane"))
        assert v._resolve_dim("Z-plane") == "z"
