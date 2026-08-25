"""
Femtonics MESc reader contract tests.

A ``.mesc`` file holds one MUnit per scan, and ``MethodType`` decides both
what axis 0 of ``Channel_N`` means and how the scanned sub-regions are packed
into the raw page. These tests build a synthetic file covering every layout
`MescArray` claims to handle, and pin the unpacking against hand-computed
expectations read straight out of h5py.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from mbo_utilities.arrays.mesc import MescArray, list_mesc_units
from mbo_utilities.reader import imread


# ============================================================
# synthetic fixture
# ============================================================

def _curve(unit, idx, name, values, delta=1.0):
    g = unit.create_group(f"Curve_{idx}")
    g.attrs["Name"] = name
    g.attrs["CurveDataXRawDelta"] = delta
    g.create_dataset(
        "CurveDataYIdxNextSample", data=np.arange(1, len(values) + 1, dtype=np.int64)
    )
    g.create_dataset("CurveDataYRawData", data=np.asarray(values))


def _protocol(pattern):
    return json.dumps(
        {
            "protocol": {"scanners": {"mainPatternIndex": 1}},
            "scanPatterns": {"patterns": [pattern]},
        }
    )


def _boxes(boxes):
    """MESc stores 1-based, lower-left/upper-right pixel corners."""
    return [
        {"lowerLeftFramePix": [c0 + 1, r0 + 1], "upperRightFramePix": [c1, r1]}
        for (r0, r1, c0, c1) in boxes
    ]


_GUIDELINE = [[[0, 1], [0, 0], [0, 0]], [[0, 1], [1, 1], [0, 0]]]


@pytest.fixture(scope="module")
def mesc_path(tmp_path_factory):
    """A .mesc with one unit per layout MescArray supports."""
    path = tmp_path_factory.mktemp("mesc") / "synthetic.mesc"
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        s = f.create_group("MSession_0")

        # MUnit_0 - MethodType 2 z-stack: axis 0 is depth, no time axis
        u = s.create_group("MUnit_0")
        u.attrs.update(
            {"MethodType": 2, "VecChannelsSize": 2, "TStepInMs": 33.0,
             "MeasurementDatePosix": 1_700_000_000, "Comment": "zstack"}
        )
        for c in range(2):
            u.create_dataset(
                f"Channel_{c}",
                data=(rng.integers(0, 4000, (20, 64, 48)) + c).astype(np.uint16),
            )

        # MUnit_1 - MethodType 8 chessboard: 4 ROIs tiled along X
        u = s.create_group("MUnit_1")
        u.attrs.update(
            {"MethodType": 8, "VecChannelsSize": 2, "TStepInMs": 50.0,
             "MeasurementDatePosix": 1_700_000_100, "Comment": "chessboard"}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {
                "centerPoints": np.arange(12).reshape(3, 4).tolist(),
                "pixelSizeX": 0.8,
                "rotation": {"e": [0.0, 0.0, 0.0]},
            }
        )
        for c in range(2):
            u.create_dataset(
                f"Channel_{c}",
                data=np.arange(6 * 32 * 96, dtype=np.uint16).reshape(6, 32, 96) + c,
            )

        # MUnit_2 - MethodType 9 ribbon transverse: ragged ROI boxes
        u = s.create_group("MUnit_2")
        u.attrs.update(
            {"MethodType": 9, "VecChannelsSize": 1, "TStepInMs": 25.0,
             "MeasurementDatePosix": 1_700_000_200, "Comment": "ribbon"}
        )
        u.attrs["BreakViewJSON"] = json.dumps(
            {"measurementROIMaps": _boxes([(0, 10, 0, 20), (12, 30, 0, 16)])}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {"guideLine": _GUIDELINE, "pixelSizeL": 1.5}
        )
        u.create_dataset(
            "Channel_0", data=np.arange(5 * 40 * 24, dtype=np.uint16).reshape(5, 40, 24)
        )

        # MUnit_3 - MethodType 6 linescan: 8 frames of 4 lines packed into Y
        u = s.create_group("MUnit_3")
        u.attrs.update(
            {"MethodType": 6, "VecChannelsSize": 1, "TStepInMs": 2.0,
             "MeasurementDatePosix": 1_700_000_300, "Comment": "linescan"}
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": _boxes([(0, 4, 0, 12), (0, 4, 12, 30)])}]}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {"guideLine": _GUIDELINE, "pixelSize": 0.5}
        )
        u.create_dataset(
            "Channel_0", data=np.arange(1 * 32 * 30, dtype=np.uint16).reshape(1, 32, 30)
        )
        _curve(u, 0, "DiI2", [1, 0, 1], delta=4.0)

        # MUnit_4 - MethodType 1 plain timeseries
        u = s.create_group("MUnit_4")
        u.attrs.update(
            {"MethodType": 1, "VecChannelsSize": 1, "TStepInMs": 100.0,
             "MeasurementDatePosix": 1_700_000_400, "Comment": "timeseries"}
        )
        u.create_dataset(
            "Channel_0", data=np.arange(7 * 16 * 18, dtype=np.uint16).reshape(7, 16, 18)
        )

        # MUnit_5 - MethodType 7 multiline with dichroic light-path switching
        u = s.create_group("MUnit_5")
        u.attrs.update(
            {"MethodType": 7, "VecChannelsSize": 2, "TStepInMs": 3.0,
             "MeasurementDatePosix": 1_700_000_500, "Comment": "dichro"}
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": _boxes([(0, 2, 0, 10)])}]}
        )
        u.attrs["MultiROIProtocolJSON"] = _protocol(
            {"centerPoints": np.arange(9).reshape(3, 3).tolist(), "pixelSize": 0.25}
        )
        for c in range(2):
            u.create_dataset(
                f"Channel_{c}",
                data=np.arange(1 * 20 * 10, dtype=np.uint16).reshape(1, 20, 10)
                + c * 1000,
            )
        _curve(u, 0, "DichroSw_AO1", [1, 2] * 5)
    return path


@pytest.fixture
def raw(mesc_path):
    """Direct h5py access to the fixture, for expectation-building."""
    with h5py.File(mesc_path, "r") as f:
        yield f


# ============================================================
# discovery + dispatch
# ============================================================

def test_list_units_reports_every_layout(mesc_path):
    units = list_mesc_units(mesc_path)
    assert [u["munit"] for u in units] == [f"MUnit_{i}" for i in range(6)]
    assert [u["kind"] for u in units] == [
        "zstack", "tiled", "boxes", "packed", "frames", "packed",
    ]
    assert [u["shape"] for u in units] == [
        (1, 2, 20, 64, 48),
        (6, 2, 4, 32, 24),
        (5, 1, 2, 18, 20),
        (8, 1, 2, 4, 18),
        (7, 1, 1, 16, 18),
        (5, 2, 1, 2, 10),
    ]


def test_imread_dispatches_to_mesc_array(mesc_path):
    arr = imread(mesc_path)
    assert isinstance(arr, MescArray)
    # no explicit unit -> the file's first unit, deterministically
    assert arr.unit_key == "MSession_0/MUnit_0"
    assert arr.reader_kwargs == {"unit": "MSession_0/MUnit_0"}


@pytest.mark.parametrize("selector", [2, "MUnit_2", "MSession_0/MUnit_2"])
def test_unit_selectors_are_equivalent(mesc_path, selector):
    assert MescArray(mesc_path, unit=selector).unit_key == "MSession_0/MUnit_2"


@pytest.mark.parametrize("selector", [99, -1, "MUnit_77", "MSession_9/MUnit_0"])
def test_bad_unit_selector_raises(mesc_path, selector):
    with pytest.raises(ValueError, match="unit"):
        MescArray(mesc_path, unit=selector)


# ============================================================
# per-layout unpacking
# ============================================================

def test_zstack_axis0_is_depth(mesc_path, raw):
    arr = MescArray(mesc_path, unit=0)
    src = raw["MSession_0/MUnit_0/Channel_1"][:]
    assert arr.shape == (1, 2, 20, 64, 48)
    assert arr.metadata["mesc_z_axis_meaning"] == "depth"
    assert arr.metadata["dz"] is None  # MESc records no trustworthy z-step
    assert arr.num_rois == 1
    assert np.array_equal(arr[0, 1, 7], src[7])
    assert np.array_equal(arr[0, 1, 3:6], src[3:6])


def test_chessboard_rois_tile_along_x_and_flip_y(mesc_path, raw):
    arr = MescArray(mesc_path, unit=1)
    src = raw["MSession_0/MUnit_1/Channel_0"][:]
    assert arr.shape == (6, 2, 4, 32, 24)
    assert arr.num_rois == 4
    assert arr.metadata["mesc_flip_y"] is True
    for r in range(4):
        expected = src[:, :, r * 24 : (r + 1) * 24][:, ::-1, :]
        assert np.array_equal(arr[:, 0, r], expected), f"roi {r}"


def test_ribbon_boxes_are_padded_to_the_largest_roi(mesc_path, raw):
    arr = MescArray(mesc_path, unit=2)
    src = raw["MSession_0/MUnit_2/Channel_0"][:]
    assert arr.shape == (5, 1, 2, 18, 20)

    padded = np.zeros((5, 18, 20), dtype=src.dtype)
    padded[:, :10, :20] = src[:, 0:10, 0:20]
    assert np.array_equal(arr[:, 0, 0], padded[:, ::-1, :])

    padded = np.zeros((5, 18, 20), dtype=src.dtype)
    padded[:, :18, :16] = src[:, 12:30, 0:16]
    assert np.array_equal(arr[:, 0, 1], padded[:, ::-1, :])

    # the padding is recorded so a consumer can mask it out
    extents = arr.metadata["mesc_roi_extents"]
    assert (extents[0]["height"], extents[0]["y_start"]) == (10, 8)
    assert (extents[1]["height"], extents[1]["y_start"]) == (18, 0)


def test_linescan_frames_unpack_from_the_y_axis(mesc_path, raw):
    arr = MescArray(mesc_path, unit=3)
    src = raw["MSession_0/MUnit_3/Channel_0"][:]
    assert arr.shape == (8, 1, 2, 4, 18)
    for t in range(8):
        assert np.array_equal(arr[t, 0, 1], src[0, t * 4 : (t + 1) * 4, 12:30])
    # out-of-order timepoints must not be silently sorted
    assert np.array_equal(
        arr[[5, 1], 0, 1],
        np.stack([src[0, 20:24, 12:30], src[0, 4:8, 12:30]]),
    )


def test_dichroic_multiline_splits_light_paths_across_channels(mesc_path, raw):
    arr = MescArray(mesc_path, unit=5)
    green = raw["MSession_0/MUnit_5/Channel_0"][:]
    red = raw["MSession_0/MUnit_5/Channel_1"][:]
    # 10 source frames alternating green/red -> 5 paired timepoints
    assert arr.shape == (5, 2, 1, 2, 10)
    assert arr.metadata["mesc_dichroic"] is True
    assert arr.metadata["channel_names"] == ["Green", "Red"]
    for t in range(5):
        assert np.array_equal(arr[t, 0, 0], green[0, 4 * t : 4 * t + 2, 0:10])
        assert np.array_equal(arr[t, 1, 0], red[0, 4 * t + 2 : 4 * t + 4, 0:10])


def test_timeseries_indexes_like_a_plain_stack(mesc_path, raw):
    arr = MescArray(mesc_path, unit=4)
    src = raw["MSession_0/MUnit_4/Channel_0"][:]
    assert arr.shape == (7, 1, 1, 16, 18)
    assert arr.metadata["mesc_z_axis_meaning"] == "none"
    assert np.array_equal(arr[2, 0, 0], src[2])
    # spatial keys are honoured, and integer keys drop their axis
    assert np.array_equal(arr[1:4, 0, 0, 2:5, ::2], src[1:4, 2:5, ::2])
    assert arr[3, 0, 0].shape == (16, 18)
    assert arr[3:4, :, :].shape == (1, 1, 1, 16, 18)


# ============================================================
# ROI interface
# ============================================================

def test_roi_selection_collapses_z_to_one_roi(mesc_path, raw):
    arr = MescArray(mesc_path, unit=1)
    src = raw["MSession_0/MUnit_1/Channel_0"][:]
    arr.roi = 3
    assert arr.shape == (6, 2, 1, 32, 24)
    assert np.array_equal(arr[:, 0, 0], src[:, :, 48:72][:, ::-1, :])


def test_roi_zero_splits_all(mesc_path):
    arr = MescArray(mesc_path, unit=1)
    arr.roi = 0
    assert list(arr.iter_rois()) == [1, 2, 3, 4]
    # click hands `--roi 0` over as a list; iter_rois expands it the same way
    arr.roi = [0]
    assert list(arr.iter_rois()) == [1, 2, 3, 4]


def test_slider_labels_match_what_the_viewer_renders(mesc_path):
    arr = MescArray(mesc_path, unit=1)
    assert arr.slider_dim_labels == ("Timepoint", "Channel", "ROI")
    # each split subplot holds a single ROI, so Z stops being scrollable
    arr.roi = 0
    assert arr.slider_dim_labels == ("Timepoint", "Channel")
    assert MescArray(mesc_path, unit=0).slider_dim_labels == ("Channel", "Z-plane")


# ============================================================
# alignment + metadata
# ============================================================

def test_sync_frame_is_reported_but_not_applied_by_default(mesc_path):
    arr = MescArray(mesc_path, unit=3)
    assert arr.metadata["mesc_sync_frame"] == 2
    assert arr.metadata["mesc_start_frame"] == 0
    assert arr.shape[0] == 8


def test_sync_key_crops_to_the_detected_frame(mesc_path, raw):
    arr = MescArray(mesc_path, unit=3, sync_key="DiI2")
    src = raw["MSession_0/MUnit_3/Channel_0"][:]
    assert arr.shape[0] == 6
    assert np.array_equal(arr[0, 0, 1], src[0, 8:12, 12:30])


def test_start_frame_beyond_the_recording_is_ignored(mesc_path):
    arr = MescArray(mesc_path, unit=4, start_frame=999)
    assert arr.shape[0] == 7


def test_dichroic_frame_rate_follows_the_timepoints_reported(mesc_path):
    """fs must describe the T axis, not the scanner.

    Switching hands each light path every other raw frame, so a channel's own
    timepoints arrive half as fast as TStepInMs ticks. Reporting the raw rate
    would make num_timepoints / fs claim half the duration recorded.
    """
    md = MescArray(mesc_path, unit=5).metadata
    assert md["mesc_light_paths"] == 2
    assert md["mesc_raw_frame_rate"] == pytest.approx(1000 / 3.0)
    assert md["fs"] == md["frame_rate"] == pytest.approx(1000 / 3.0 / 2)
    # 10 raw frames at 3 ms each is 30 ms, however the pairs are counted
    assert md["num_timepoints"] / md["fs"] == pytest.approx(0.030)

    # a unit without switching is untouched
    plain = MescArray(mesc_path, unit=2).metadata
    assert plain["mesc_light_paths"] == 1
    assert plain["fs"] == plain["mesc_raw_frame_rate"] == pytest.approx(40.0)


@pytest.mark.parametrize("unit,expected", [(1, 2), (2, 1), (5, 2)])
def test_num_color_channels_is_reported_next_to_nchannels(mesc_path, unit, expected):
    md = MescArray(mesc_path, unit=unit).metadata
    assert md["num_color_channels"] == md["nchannels"] == expected


def test_dichroic_sync_frame_is_converted_to_light_path_timepoints(tmp_path):
    """The sync search counts raw frames; the crop counts timepoints.

    TStepInMs ticks once per raw scanner frame, so `_find_sync_frame` lands on
    a raw index. One timepoint of a switched unit spans two raw frames, so
    applying that index unconverted would crop twice as much as the edge asks.
    """
    path = tmp_path / "dichro_sync.mesc"
    with h5py.File(path, "w") as f:
        u = f.create_group("MSession_0").create_group("MUnit_0")
        u.attrs.update(
            {"MethodType": 7, "VecChannelsSize": 2, "TStepInMs": 2.0,
             "MeasurementDatePosix": 1_700_000_000, "Comment": "dichro+sync"}
        )
        u.attrs["CoordinateMapJSON"] = json.dumps(
            {"maps": [{"measurementROIs": _boxes([(0, 2, 0, 10)])}]}
        )
        for c in range(2):
            u.create_dataset(
                f"Channel_{c}",
                data=np.arange(24 * 10, dtype=np.uint16).reshape(1, 24, 10) + c * 1000,
            )
        _curve(u, 0, "DichroSw_AO1", [1, 2] * 6)
        _curve(u, 1, "DiI2", [1, 0, 1], delta=8.0)

    arr = MescArray(path, unit=0)
    # falling edge at 8 ms / 2 ms = raw frame 4, which is timepoint 2
    assert arr.metadata["mesc_sync_frame_raw"] == 4
    assert arr.metadata["mesc_sync_frame"] == 2
    assert arr.shape[0] == 6  # 12 raw frames -> 6 pairs, nothing cropped yet

    cropped = MescArray(path, unit=0, sync_key="DiI2")
    assert cropped.metadata["mesc_start_frame"] == 2
    assert cropped.shape[0] == 4
    # timepoint 0 of the cropped array is the pair at raw frames 4 and 5
    assert cropped._source_frames(0, [0]) == 4
    assert cropped._source_frames(1, [0]) == 5


def test_canonical_metadata_is_populated(mesc_path):
    md = MescArray(mesc_path, unit=2).metadata
    assert md["fs"] == pytest.approx(40.0)
    assert md["dx"] == md["dy"] == pytest.approx(1.5)
    assert (md["num_timepoints"], md["num_zplanes"], md["nchannels"]) == (5, 2, 1)
    assert (md["Ly"], md["Lx"]) == (18, 20)
    assert md["num_mrois"] == 2
    assert md["mesc_modality_name"] == "ribbon_transverse"
    assert md["mesc_unit"] == "MSession_0/MUnit_2"
    # the big embedded protocol JSON is parsed, not dumped into metadata
    assert "MultiROIProtocolJSON" not in md["mesc_attrs"]
    assert len(md["mesc_centroids"]) == 2


def test_metadata_overrides_do_not_touch_the_read_only_file(mesc_path):
    arr = MescArray(mesc_path, unit=4)
    arr.metadata = {"dz": 12.0}
    assert arr.metadata["dz"] == 12.0
    assert MescArray(mesc_path, unit=4).metadata["dz"] is None


# ============================================================
# lazy-array contract
# ============================================================

def test_reads_only_the_requested_frames(mesc_path, monkeypatch):
    """A single-timepoint read must not pull the whole channel into memory."""
    arr = MescArray(mesc_path, unit=1)
    dataset = arr._channels[0]
    reads = []
    original = dataset.__class__.__getitem__
    monkeypatch.setattr(
        dataset.__class__,
        "__getitem__",
        lambda self, key: (reads.append(key), original(self, key))[1],
    )
    frame = arr[2, 0, 0]
    assert frame.shape == (32, 24)
    assert reads, "expected at least one h5py read"
    for key in reads:
        # every read is bounded: never a bare full-dataset slice
        assert key != slice(None)


def test_squeeze_and_dims_follow_the_canonical_contract(mesc_path):
    arr = MescArray(mesc_path, unit=4)
    assert arr.dims == ("T", "C", "Z", "Y", "X")
    assert arr.ndim == 5
    assert arr.squeeze().shape == (7, 16, 18)
    assert (arr.nt, arr.nc, arr.nz, arr.ny, arr.nx) == (7, 1, 1, 16, 18)


def test_astype_converts_lazily(mesc_path):
    arr = MescArray(mesc_path, unit=4).astype(np.float32)
    assert arr.dtype == np.float32
    assert arr[0, 0, 0].dtype == np.float32


def test_phase_correction_is_available_but_off(mesc_path):
    arr = MescArray(mesc_path, unit=4)
    assert hasattr(arr, "phase_correction")
    assert arr.fix_phase is False
    baseline = np.asarray(arr[0, 0, 0])
    arr.fix_phase = True
    assert arr.fix_phase is True
    corrected = np.asarray(arr[0, 0, 0])
    assert corrected.shape == baseline.shape
    assert arr.get_offset_at(0, 0, 0) is not None


def test_imwrite_round_trips_through_zarr(mesc_path, tmp_path):
    from mbo_utilities import imwrite

    arr = MescArray(mesc_path, unit=1)
    out = tmp_path / "out"
    out.mkdir()
    imwrite(arr, out, ext=".zarr", overwrite=True)
    back = imread(next(out.glob("*.zarr")))
    assert back.shape == arr.shape
    assert np.array_equal(np.asarray(back[:, :, :]), np.asarray(arr[:, :, :]))


def test_imwrite_round_trips_two_channels_through_h5(mesc_path, tmp_path):
    """Dual-PMT is the normal case here, so h5 must keep the C axis.

    A single channel still writes 4D TZYX -- that is what every existing mbo
    h5 holds -- but two channels write 5D TCZYX rather than silently reducing
    to channel 0.
    """
    from mbo_utilities import imwrite

    arr = MescArray(mesc_path, unit=1)
    assert arr.shape[1] == 2
    out = tmp_path / "dual"
    out.mkdir()
    written = imwrite(arr, out, ext=".h5", overwrite=True)

    with h5py.File(written, "r") as f:
        dset = f[next(iter(f))]
        assert dset.shape == arr.shape
        assert list(dset.attrs["dims"]) == ["T", "C", "Z", "Y", "X"]

    back = imread(written)
    assert back.shape == arr.shape
    assert np.array_equal(np.asarray(back[:, :, :]), np.asarray(arr[:, :, :]))


def test_imwrite_single_channel_h5_stays_four_dimensional(mesc_path, tmp_path):
    from mbo_utilities import imwrite

    arr = MescArray(mesc_path, unit=2)
    assert arr.shape[1] == 1
    out = tmp_path / "single"
    out.mkdir()
    written = imwrite(arr, out, ext=".h5", overwrite=True)

    with h5py.File(written, "r") as f:
        dset = f[next(iter(f))]
        assert dset.shape == (arr.shape[0], *arr.shape[2:])
        assert list(dset.attrs["dims"]) == ["T", "Z", "Y", "X"]

    assert imread(written).shape == arr.shape


def test_imwrite_roi_zero_fans_out_one_directory_per_roi(mesc_path, tmp_path):
    from mbo_utilities import imwrite

    arr = MescArray(mesc_path, unit=1)
    arr.roi = 0
    out = tmp_path / "split"
    out.mkdir()
    imwrite(arr, out, ext=".tiff", overwrite=True)
    assert sorted(p.name for p in out.iterdir()) == [
        "roi01", "roi02", "roi03", "roi04",
    ]


# ============================================================
# launch picker
# ============================================================

@pytest.fixture(scope="module")
def single_unit_mesc(tmp_path_factory):
    """A .mesc holding exactly one MUnit."""
    path = tmp_path_factory.mktemp("mesc_one") / "one.mesc"
    with h5py.File(path, "w") as f:
        u = f.create_group("MSession_0").create_group("MUnit_0")
        u.attrs.update({"MethodType": 1, "VecChannelsSize": 1, "TStepInMs": 50.0})
        u.create_dataset(
            "Channel_0", data=np.zeros((4, 8, 8), dtype=np.uint16)
        )
    return path


class TestUnitPicker:
    """`.mesc` always asks which unit to open — it is never a safe default."""

    @staticmethod
    def _patch(monkeypatch, result):
        from mbo_utilities.gui import run_gui as rg

        calls = []

        def _fake(path, units):
            calls.append((Path(path).name, len(units)))
            return result

        monkeypatch.setattr(rg, "_prompt_for_mesc_unit", _fake)
        return calls

    def test_prompts_even_when_the_file_holds_one_unit(
        self, single_unit_mesc, monkeypatch
    ):
        from mbo_utilities.gui.run_gui import _resolve_mesc_unit

        calls = self._patch(monkeypatch, "MSession_0/MUnit_0")
        kwargs, proceed = _resolve_mesc_unit(single_unit_mesc, None)
        assert calls == [("one.mesc", 1)]
        assert (kwargs, proceed) == ({"unit": "MSession_0/MUnit_0"}, True)

    def test_explicit_unit_bypasses_the_picker(self, mesc_path, monkeypatch):
        from mbo_utilities.gui.run_gui import _resolve_mesc_unit

        calls = self._patch(monkeypatch, "MSession_0/MUnit_0")
        assert _resolve_mesc_unit(mesc_path, 3) == ({"unit": 3}, True)
        assert calls == []

    def test_cancelling_aborts_instead_of_opening_something(
        self, mesc_path, monkeypatch
    ):
        from mbo_utilities.gui.run_gui import _resolve_mesc_unit

        self._patch(monkeypatch, None)
        assert _resolve_mesc_unit(mesc_path, None) == ({}, False)

    def test_no_qt_falls_through_to_the_first_unit(self, mesc_path, monkeypatch):
        from mbo_utilities.gui import run_gui as rg

        self._patch(monkeypatch, rg._PICKER_UNAVAILABLE)
        assert rg._resolve_mesc_unit(mesc_path, None) == ({}, True)

    def test_non_mesc_inputs_are_left_alone(self, tmp_path, monkeypatch):
        from mbo_utilities.gui.run_gui import _resolve_mesc_unit

        calls = self._patch(monkeypatch, "MSession_0/MUnit_0")
        other = tmp_path / "scan.tif"
        other.touch()
        assert _resolve_mesc_unit(other, None) == ({}, True)
        assert _resolve_mesc_unit(tmp_path, None) == ({}, True)
        assert calls == []


# ============================================================
# viewer fit + unit widget
# ============================================================

class TestViewerFit:
    """Every non-spatial axis is a slider — no ROI pinning, no subplot fan-out."""

    def test_three_scrollable_axes_stay_one_array(self, mesc_path):
        arr = MescArray(mesc_path, unit=1)  # T=6, C=2, R=4
        assert arr.roi is None
        assert arr.shape == (6, 2, 4, 32, 24)
        assert arr.slider_dim_labels == ("Timepoint", "Channel", "ROI")

    @pytest.mark.parametrize("unit", [0, 2, 3, 4, 5])
    def test_every_unit_reports_one_label_per_scrollable_axis(self, mesc_path, unit):
        from mbo_utilities.gui.widgets.mesc_units import display_wrap

        arr = MescArray(mesc_path, unit=unit)
        assert arr.roi is None
        view = display_wrap(arr)
        assert len(view.shape) - 2 == len(arr.slider_dim_labels), arr.unit_key

    def test_roi_axis_is_labelled_roi_not_z(self, mesc_path):
        # a ribbon unit's Z axis holds ROIs; a real z-stack's holds depth
        assert MescArray(mesc_path, unit=2).slider_dim_labels == (
            "Timepoint", "ROI",
        )
        assert MescArray(mesc_path, unit=0).slider_dim_labels == (
            "Channel", "Z-plane",
        )

    def test_mesc_array_is_found_through_the_display_wrappers(self, mesc_path):
        from mbo_utilities.gui.widgets.mesc_units import display_wrap, mesc_array_of

        arr = MescArray(mesc_path, unit=1)
        assert mesc_array_of(display_wrap(arr)) is arr
        # a squeeze layer is only inserted when a singleton axis exists
        arr.roi = 1
        assert mesc_array_of(display_wrap(arr)) is arr
        assert mesc_array_of(np.zeros((3, 3))) is None


class TestUnitWidgetSupport:
    def test_supported_only_for_mesc_backed_viewers(self, mesc_path):
        from mbo_utilities.gui.widgets.mesc_units import (
            MescUnitsWidget,
            display_wrap,
        )

        class FakeIW:
            def __init__(self, data):
                self.data = data

        class FakeParent:
            def __init__(self, data):
                self.image_widget = FakeIW(data)

        arr = MescArray(mesc_path, unit=4)
        assert MescUnitsWidget.is_supported(FakeParent([display_wrap(arr)]))
        assert not MescUnitsWidget.is_supported(FakeParent([np.zeros((4, 4, 4))]))
        assert not MescUnitsWidget.is_supported(FakeParent([]))
        assert not MescUnitsWidget.is_supported(FakeParent(None))


def test_unit_switching_stands_down_on_split_roi_views(mesc_path):
    """`--roi 0` fans ROIs across subplots; swapping would strand all but one."""
    from mbo_utilities.gui.widgets.mesc_units import MescUnitsWidget, display_wrap

    arr = MescArray(mesc_path, unit=1, roi=0)
    views = [display_wrap(arr) for _ in range(arr.num_rois)]

    drawn = []

    class FakeIW:
        data = views

        class figure:  # noqa: N801 - stands in for the fastplotlib figure
            pass

    class FakeParent:
        image_widget = FakeIW()
        logger = None

    widget = MescUnitsWidget(FakeParent())
    assert widget.is_supported(FakeParent())

    # draw() must bail before touching any combo state for a multi-subplot view
    import mbo_utilities.gui.widgets.mesc_units as mod

    class _Recorder:
        def __getattr__(self, name):
            def _call(*args, **kwargs):
                drawn.append(name)
                if name == "combo":
                    raise AssertionError("combo drawn for a split-ROI view")
                if name == "get_content_region_avail":
                    return type("V", (), {"x": 100.0})()
                return None

            return _call

    original = mod.imgui
    mod.imgui = _Recorder()
    try:
        widget.draw()
    finally:
        mod.imgui = original
    assert "text_disabled" in drawn
