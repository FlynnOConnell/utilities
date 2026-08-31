"""Temporal frame averaging: the read-time view, and the viewer lock that
installs it as a pipeline step."""

import numpy as np
import pytest

from mbo_utilities.arrays import FrameAveragedView, average_frames
from mbo_utilities.arrays.numpy import NumpyArray

FIGURE_SIZE = (900, 700)


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    return (rng.random((22, 2, 1, 4, 5)) * 500).astype(np.int16)


@pytest.fixture
def source(raw):
    return NumpyArray(raw, dims="TCZYX", metadata={"fs": 30.0, "num_frames": 22})


def reference(raw, factor):
    """What numpy would give: whole bins only, rounded back to the dtype."""
    n = raw.shape[0] // factor
    mean = raw[: n * factor].reshape(n, factor, *raw.shape[1:]).mean(axis=1)
    return np.rint(mean).astype(raw.dtype)


class TestShape:
    def test_bins_are_whole_and_the_tail_is_dropped(self, source, raw):
        view = average_frames(source, 4)
        assert view.shape == (5, 2, 1, 4, 5)  # 22 // 4, the last 2 frames dropped
        assert view.nt == 5 and len(view) == 5 and view.ndim == 5
        assert view.dims == ("T", "C", "Z", "Y", "X")

    def test_factor_one_is_the_source_itself(self, source):
        assert average_frames(source, 1) is source

    def test_wrapping_a_view_replaces_it_rather_than_stacking(self, source):
        once = average_frames(source, 2)
        twice = average_frames(once, 4)
        assert twice.source is source and twice.factor == 4
        assert average_frames(once, 1) is source

    def test_a_factor_past_the_end_is_refused(self, source):
        with pytest.raises(ValueError):
            average_frames(source, 100)

    def test_zero_and_negative_are_refused(self, source):
        for bad in (0, -3):
            with pytest.raises(ValueError):
                FrameAveragedView(source, bad)


class TestReads:
    """Every key form must agree with numpy on the binned array."""

    @pytest.fixture
    def pair(self, source, raw):
        return average_frames(source, 4), reference(raw, 4)

    def test_whole_array(self, pair):
        view, ref = pair
        assert np.array_equal(np.asarray(view[:]), ref)
        assert np.array_equal(np.asarray(view), ref)

    def test_integer_and_negative_t(self, pair):
        view, ref = pair
        assert np.array_equal(view[2], ref[2])
        assert np.array_equal(view[-1], ref[-1])

    def test_slices_and_steps(self, pair):
        view, ref = pair
        assert np.array_equal(view[1:4], ref[1:4])
        assert np.array_equal(view[0:5:2], ref[0:5:2])
        assert view[3:3].shape == (0, 2, 1, 4, 5)

    def test_fancy_t(self, pair):
        view, ref = pair
        assert np.array_equal(view[[3, 1]], ref[[3, 1]])

    def test_spatial_subkey_reads_only_that_crop(self, pair):
        view, ref = pair
        assert np.array_equal(view[1:3, 0, 0, 1:3, 2:4], ref[1:3, 0, 0, 1:3, 2:4])

    def test_t_out_of_range(self, pair):
        view, _ = pair
        with pytest.raises(IndexError):
            view[99]

    def test_the_cache_does_not_change_what_is_read(self, pair):
        view, ref = pair
        for _ in range(3):
            assert np.array_equal(view[2], ref[2])


class TestDtype:
    def test_source_dtype_is_preserved_by_default(self, source, raw):
        view = average_frames(source, 4)
        assert view.dtype == raw.dtype
        assert np.asarray(view[:]).dtype == raw.dtype

    def test_float32_keeps_the_fractional_means(self, source, raw):
        view = average_frames(source, 4, dtype="float32")
        assert view.dtype == np.float32
        exact = raw[:20].reshape(5, 4, 2, 1, 4, 5).mean(axis=1, dtype=np.float32)
        np.testing.assert_allclose(np.asarray(view[:]), exact, rtol=1e-6)

    def test_an_unknown_dtype_is_refused(self, source):
        with pytest.raises(ValueError):
            FrameAveragedView(source, 2, dtype="int8")


class TestMetadata:
    """Averaging N frames divides the frame rate by N. Every downstream
    window, detrend and trace axis is in seconds, so this must be right."""

    def test_rate_and_frame_count_are_scaled(self, source):
        meta = average_frames(source, 4).metadata
        assert meta["fs"] == pytest.approx(7.5)
        assert meta["num_frames"] == 5
        assert meta["frame_average"] == 4

    def test_every_rate_alias_is_scaled(self, raw):
        arr = NumpyArray(raw, dims="TCZYX", metadata={"frame_rate": 30.0, "framerate": 30.0})
        meta = average_frames(arr, 2).metadata
        assert meta["frame_rate"] == pytest.approx(15.0)
        assert meta["framerate"] == pytest.approx(15.0)

    def test_history_records_the_step(self, source):
        history = average_frames(source, 4).metadata["processing_history"]
        assert {"step": "frame_average", "factor": 4} in history

    def test_every_registered_alias_is_retimed(self, raw):
        """A single alias left claiming the original rate makes the resolver
        warn about a stale alias, and anything reading `fps` or `dt` straight
        from the dict gets the pre-binning number."""
        from mbo_utilities.metadata import get_param

        meta = {
            "fs": 30.0, "fps": 30.0, "fr": 30.0, "scanFrameRate": 30.0,
            "frameRate": 30.0, "sampling_frequency": 30.0, "frame_rate_hz": 30.0,
            "finterval": 1 / 30, "dt": 1 / 30, "frame_period": 1 / 30,
        }
        out = average_frames(NumpyArray(raw, dims="TCZYX", metadata=meta), 3).metadata
        for key in ("fs", "fps", "fr", "scanFrameRate", "frameRate",
                    "sampling_frequency", "frame_rate_hz"):
            assert out[key] == pytest.approx(10.0), key
        for key in ("finterval", "dt", "frame_period"):
            assert out[key] == pytest.approx(0.1), key
        assert get_param(out, "fs") == pytest.approx(10.0)

    def test_missing_rate_is_left_alone(self, raw):
        meta = average_frames(NumpyArray(raw, dims="TCZYX"), 2).metadata
        assert "fs" not in meta and meta["frame_average"] == 2


class TestPassthrough:
    def test_domain_attributes_forward_to_the_source(self, source, tmp_path):
        view = average_frames(source, 4)
        assert view.source is source
        assert view._arr is source
        # a name only the source defines still resolves
        source.some_reader_detail = "kept"
        assert view.some_reader_detail == "kept"

    def test_the_frame_count_never_leaks_from_the_source(self, source):
        view = average_frames(source, 4)
        # nt/num_frames/len come off _shape5d, not the wrapped reader
        assert view.nt == 5 and len(view) == 5
        assert view.shape[0] == 5 and source.shape[0] == 22

    def test_imwrite_bakes_the_averaging_in(self, source, raw, tmp_path):
        from mbo_utilities.reader import imread

        view = average_frames(source, 4)
        view._imwrite(tmp_path, ext=".tiff", overwrite=True)
        written = list(tmp_path.rglob("*.tif*"))
        assert written, "nothing written"
        back = np.asarray(imread(written[0])[:]).squeeze()
        assert back.shape[0] == 5
        np.testing.assert_allclose(
            back.reshape(5, -1), reference(raw, 4).squeeze().reshape(5, -1), atol=1
        )


class TestViewerLock:
    """The GUI side: ticking "Apply to dataset" swaps the viewer onto the
    view, and everything downstream reads the averaged frames."""

    @staticmethod
    def _gui(nt=120, fs=30.0):
        from mbo_utilities.gui.run_gui import _create_image_widget
        from mbo_utilities.gui.widgets.preview_data import PreviewDataWidget

        rng = np.random.default_rng(0)
        data = (rng.random((nt, 1, 1, 64, 64)) * 100).astype(np.int16)
        iw = _create_image_widget(
            NumpyArray(data, dims="TCZYX", metadata={"fs": fs}),
            widget="preview",
            figure_kwargs_override={"size": FIGURE_SIZE},
        )
        gui = next(
            w for w in iw.figure.imgui_windows.values()
            if isinstance(w, PreviewDataWidget)
        )
        return iw, gui

    def test_locking_bins_the_data_and_resets_the_window(self):
        iw, gui = self._gui()
        try:
            gui.window_size = 10
            gui.frame_average = gui.window_size
            assert gui.frame_average == 10
            assert gui.shape[0] == 12
            assert gui.window_size == 1, "the size is spent; it now applies on top"
            assert gui._get_data_arrays()[0].metadata["fs"] == pytest.approx(3.0)
        finally:
            iw.close()

    def test_unlocking_restores_the_source(self):
        iw, gui = self._gui()
        try:
            gui.frame_average = 10
            gui.frame_average = 1
            assert gui.shape[0] == 120
            assert gui._get_data_arrays()[0].metadata["fs"] == pytest.approx(30.0)
            assert gui._frame_average_source is None
        finally:
            iw.close()

    def test_relocking_does_not_compound(self):
        iw, gui = self._gui()
        try:
            gui.frame_average = 10
            gui.frame_average = 4
            assert gui.shape[0] == 30, "4 bins of the raw 120, not of the 12"
        finally:
            iw.close()

    def test_a_factor_past_the_end_clamps(self):
        """Averaging everything into one frame is allowed; it degrades to the
        mean image (T is gone, so the viewer squeezes it away) and unlocking
        brings the movie back."""
        iw, gui = self._gui(nt=8)
        try:
            gui.frame_average = 999
            assert gui.frame_average == 8
            assert gui.shape == (64, 64)
            gui.frame_average = 1
            assert gui.shape[0] == 8
        finally:
            iw.close()

    def test_rois_and_their_traces_follow_the_binning(self):
        import time

        from mbo_utilities.gui.widgets.widget_toggles import set_widget_enabled

        set_widget_enabled("manual_roi", True, persist=False)
        iw, gui = self._gui()
        try:
            gui.sync_manual_roi(True)
            gui.frame_average = 10
            roi = gui.manual_roi
            assert roi is not None, "the ROI widget must survive the swap"
            roi.add_roi([(10.0, 10.0), (25.0, 10.0), (25.0, 25.0), (10.0, 25.0)])
            assert roi.movie().shape[0] == 12, "runs read the averaged movie"
            roi.quick_trace(0)
            deadline = time.time() + 30
            while time.time() < deadline and not roi.has_traces():
                roi._poll_jobs()
                iw.figure.canvas.draw()
                time.sleep(0.02)
            key = roi._sorted_trace_rows()[0]
            entry = roi._trace_entry(key)
            assert entry["F"].size == 12
            assert entry["frame_average"] == 10

            # the ROI itself is (Z, Y, X) and must survive unlocking
            gui.frame_average = 1
            assert gui.manual_roi.n_rois == 1
        finally:
            set_widget_enabled("manual_roi", False, persist=False)
            iw.close()

    def test_the_trace_plot_follows_the_window_function(self):
        from mbo_utilities.gui.manual_roi import ManualRoiWidget

        iw, gui = self._gui()
        try:
            gui.sync_manual_roi(True)
            roi = gui.manual_roi
            assert isinstance(roi, ManualRoiWidget)
            assert roi._window_spec() == ("mean", 1)
            y = np.zeros(50, np.float32)
            y[25] = 10.0
            assert np.array_equal(roi._windowed(y), y), "size 1 is the raw trace"
            gui.window_size = 5
            gui.proj = "max"
            assert roi._window_spec() == ("max", 5)
            smoothed = roi._windowed(y)
            assert smoothed.shape == y.shape
            assert smoothed[23:28].tolist() == [10.0] * 5, "max spreads the peak"
        finally:
            iw.close()
